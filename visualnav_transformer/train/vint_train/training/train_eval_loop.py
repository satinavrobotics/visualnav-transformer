import itertools
import os
from typing import Dict, Optional
import gc
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import tqdm
import mlflow
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from prettytable import PrettyTable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from visualnav_transformer.train.vint_train.data.dataset.data_utils import (
    VISUALIZATION_IMAGE_SIZE,
)
from visualnav_transformer.train.vint_train.logging.logger import Logger
from visualnav_transformer.train.vint_train.logging.cli_formatter import (
    Colors, Symbols, print_info, print_success, print_section,
    format_number, format_time, create_custom_tqdm_format, TrainingProgressTracker
)
from visualnav_transformer.train.vint_train.logging.enhanced_logger import (
    display_epoch_summary
)
from visualnav_transformer.train.vint_train.visualizing.action_utils import (
    plot_trajs_and_points,
    visualize_traj_pred,
)
from visualnav_transformer.train.vint_train.visualizing.distance_utils import (
    visualize_dist_pred,
)
from visualnav_transformer.train.vint_train.visualizing.visualize_utils import (
    from_numpy,
    to_numpy,
)
from visualnav_transformer.train.vint_train.training.train_utils import (
    _log_data,
    action_reduce,
    visualize_diffusion_action_distribution,
    get_data_stats,
    normalize_data,
    unnormalize_data,
    get_delta,
    get_action,
    model_output,
    ACTION_STATS,
    setup_loggers,
    process_batch_data,
    compute_pose_loss,
    compute_distance_loss,
    cleanup_memory,
    handle_visualization,
    _compute_losses_nomad,
)


def train_nomad(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    alpha: float = 1e-4,
    beta: float = 1e-4,
    print_log_freq: int = 100,
    mlflow_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_mlflow: bool = True,
    viz_dataloader: Optional[DataLoader] = None,  # Add visualization dataloader parameter
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of distance loss
        beta: weight of pose loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_mlflow: whether to use mlflow
    """

    model.train()
    num_batches = len(dataloader)
    loggers = setup_loggers("train", print_log_freq)
    # Note: gc_pose_loss is already included in the enhanced logger
    using_prebuilt_dino = viz_dataloader is not None
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    # Enhanced tqdm with custom formatting
    desc = f"{Symbols.TRAIN} {Colors.BRIGHT_BLUE}Training Epoch {epoch+1}{Colors.RESET}"
    with tqdm.tqdm(
        dataloader,
        desc=desc,
        leave=False,
        ncols=100,
        bar_format=f"{desc}: {{percentage:3.0f}}%|{{bar}}| {{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]",
        colour='blue'
    ) as tepoch:
        for i, data in enumerate(tepoch):
            # Process batch data
            batch_data = process_batch_data(data, device, transform, using_prebuilt_dino)
            batch_obs_images = batch_data['batch_obs_images']
            batch_goal_images = batch_data['batch_goal_images']
            batch_viz_obs_images = batch_data['batch_viz_obs_images']
            batch_viz_goal_images = batch_data['batch_viz_goal_images']
            actions = batch_data['actions']
            distance = batch_data['distance']
            goal_pos = batch_data['goal_pos']
            dataset_idx = batch_data['dataset_idx']
            action_mask = batch_data['action_mask']

            B = actions.shape[0]

            # Generate random goal mask
            goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            obsgoal_cond = model(
                "vision_encoder",
                obs_img=batch_obs_images,
                goal_img=batch_goal_images,
                input_goal_mask=goal_mask,
            )

            # Prepare action data
            deltas = get_delta(actions)
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(ndeltas).to(device)
            #assert naction.shape[-1] == 2, "action dim must be 2"

            # Compute losses
            dist_loss = compute_distance_loss(model, obsgoal_cond, distance, goal_mask, device)
            pose_loss = compute_pose_loss(model, obsgoal_cond, goal_pos, goal_mask, device)

            # Sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)
            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (B,), device=device
            ).long()
            # Add noise to the clean images according to the noise magnitude at each diffusion iteration
            noisy_action = noise_scheduler.add_noise(naction, noise, timesteps)
            # Predict the noise residual
            noise_pred = model(
                "noise_pred_net",
                sample=noisy_action,
                timestep=timesteps,
                global_cond=obsgoal_cond,
            )

            # L2 loss
            diffusion_loss = action_reduce(
                F.mse_loss(noise_pred, noise, reduction="none"), action_mask
            )

            # Total loss
            loss = alpha * dist_loss + beta * pose_loss + (1 - alpha - beta) * diffusion_loss

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Exponential Moving Average of the model weights
            ema_model.step(model)

            # Logging with enhanced display
            loss_cpu = loss.item()
            dist_loss_cpu = dist_loss.item()
            pose_loss_cpu = pose_loss.item()
            diffusion_loss_cpu = diffusion_loss.item()

            # Enhanced postfix with formatted numbers and colors
            postfix_dict = {
                'total': f"{format_number(loss_cpu, 3)}",
                'dist': f"{format_number(dist_loss_cpu, 3)}",
                'pose': f"{format_number(pose_loss_cpu, 3)}",
                'diff': f"{format_number(diffusion_loss_cpu, 3)}"
            }
            tepoch.set_postfix(postfix_dict)
            # Use epoch*num_batches + i to ensure step increases continuously across epochs
            global_step = epoch * num_batches + i
            mlflow.log_metric("total_loss", loss_cpu, step=global_step)
            mlflow.log_metric("dist_loss", dist_loss.item(), step=global_step)
            mlflow.log_metric("pose_loss", pose_loss.item(), step=global_step)
            mlflow.log_metric("diffusion_loss", diffusion_loss.item(), step=global_step)

            if i % print_log_freq == 0:
                losses = _compute_losses_nomad(
                    ema_model.averaged_model,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    distance.to(device),
                    actions.to(device),
                    device,
                    action_mask.to(device),
                    goal_pos.to(device),
                )

                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())

                # Use enhanced logger display instead of legacy logging
                loggers.display_metrics(epoch, i, num_batches, use_latest=True)

                data_log = {}
                for key, logger in loggers.all_loggers.items():
                    data_log[logger.full_name()] = logger.latest()

                if use_mlflow and i % mlflow_log_freq == 0 and mlflow_log_freq != 0:
                    for k, v in data_log.items():
                        # Additional safety check for NaN/infinite values before MLflow logging
                        if not np.isnan(v) and np.isfinite(v):
                            # Use epoch*num_batches + i to ensure step increases continuously across epochs
                            global_step = epoch * num_batches + i
                            mlflow.log_metric(k, v, step=global_step)
                        else:
                            print(f"Warning: Skipping MLflow logging for {k}={v} (NaN/infinite value)")

            if image_log_freq != 0 and i % image_log_freq == 0:
                handle_visualization(
                    using_prebuilt_dino, viz_dataloader, ema_model.averaged_model,
                    noise_scheduler, batch_obs_images, batch_goal_images,
                    batch_viz_obs_images, batch_viz_goal_images, actions, distance,
                    goal_pos, device, "train", project_folder, epoch, num_images_log,
                    use_mlflow, i, num_batches, loggers, dataset_idx, mlflow_log_freq,
                    print_log_freq, image_log_freq, True
                )

            # Memory cleanup
            if i % 10 == 0:  # Every 10 batches
                cleanup_memory()

    # Display epoch summary
    display_epoch_summary(loggers, epoch)


def evaluate_nomad(
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    mlflow_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_mlflow: bool = True,
    viz_dataloader: Optional[DataLoader] = None,  # Add visualization dataloader parameter
):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        ema_model (nn.Module): exponential moving average version of model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        noise_scheduler: noise scheduler to evaluate with
        project_folder (string): path to project folder
        epoch (int): current epoch
        print_log_freq (int): how often to print logs
        mlflow_log_freq (int): how often to log to mlflow
        image_log_freq (int): how often to log images
        alpha (float): weight for action loss
        num_images_log (int): number of images to log
        eval_fraction (float): fraction of data to use for evaluation
        use_mlflow (bool): whether to use mlflow for logging
    """
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    ema_model = ema_model.averaged_model
    ema_model.eval()

    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)
    loggers = setup_loggers(eval_type, print_log_freq)
    using_prebuilt_dino = viz_dataloader is not None
    # Enhanced evaluation tqdm with custom formatting
    desc = f"{Symbols.EVAL} {Colors.BRIGHT_MAGENTA}Evaluating {eval_type.upper()} - Epoch {epoch+1}{Colors.RESET}"
    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        desc=desc,
        leave=False,
        ncols=100,
        bar_format=f"{desc}: {{percentage:3.0f}}%|{{bar}}| {{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]",
        colour='magenta'
    ) as tepoch:
        for i, data in enumerate(tepoch):
            # Process batch data
            batch_data = process_batch_data(data, device, transform, using_prebuilt_dino)
            batch_obs_images = batch_data['batch_obs_images']
            batch_goal_images = batch_data['batch_goal_images']
            batch_viz_obs_images = batch_data['batch_viz_obs_images']
            batch_viz_goal_images = batch_data['batch_viz_goal_images']
            actions = batch_data['actions']
            distance = batch_data['distance']
            goal_pos = batch_data['goal_pos']
            dataset_idx = batch_data['dataset_idx']
            action_mask = batch_data['action_mask']

            B = actions.shape[0]

            # Generate random goal mask
            rand_goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            goal_mask = torch.ones_like(rand_goal_mask).long().to(device)
            no_mask = torch.zeros_like(rand_goal_mask).long().to(device)

            rand_mask_cond = ema_model(
                "vision_encoder",
                obs_img=batch_obs_images,
                goal_img=batch_goal_images,
                input_goal_mask=rand_goal_mask,
            )

            obsgoal_cond = ema_model(
                "vision_encoder",
                obs_img=batch_obs_images,
                goal_img=batch_goal_images,
                input_goal_mask=no_mask,
            )
            obsgoal_cond = obsgoal_cond.flatten(start_dim=1)

            goal_mask_cond = ema_model(
                "vision_encoder",
                obs_img=batch_obs_images,
                goal_img=batch_goal_images,
                input_goal_mask=goal_mask,
            )

            deltas = get_delta(actions)
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(ndeltas).to(device)
            #assert naction.shape[-1] == 2, "action dim must be 2"

            # Sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (B,), device=device
            ).long()

            noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

            ### RANDOM MASK ERROR ###
            # Predict the noise residual
            rand_mask_noise_pred = ema_model(
                "noise_pred_net",
                sample=noisy_actions,
                timestep=timesteps,
                global_cond=rand_mask_cond,
            )

            # L2 loss
            rand_mask_loss = nn.functional.mse_loss(rand_mask_noise_pred, noise)

            ### NO MASK ERROR ###
            # Predict the noise residual
            no_mask_noise_pred = ema_model(
                "noise_pred_net",
                sample=noisy_actions,
                timestep=timesteps,
                global_cond=obsgoal_cond,
            )

            # L2 loss
            no_mask_loss = nn.functional.mse_loss(no_mask_noise_pred, noise)

            ### GOAL MASK ERROR ###
            # predict the noise residual
            goal_mask_noise_pred = ema_model(
                "noise_pred_net",
                sample=noisy_actions,
                timestep=timesteps,
                global_cond=goal_mask_cond,
            )

            # L2 loss
            goal_mask_loss = nn.functional.mse_loss(goal_mask_noise_pred, noise)

            # Compute pose losses for evaluation
            rand_mask_pose_loss = compute_pose_loss(ema_model, rand_mask_cond, goal_pos, rand_goal_mask, device)
            no_mask_pose_loss = compute_pose_loss(ema_model, obsgoal_cond, goal_pos, no_mask, device)
            goal_mask_pose_loss = compute_pose_loss(ema_model, goal_mask_cond, goal_pos, goal_mask, device)

            # Logging with enhanced display
            loss_cpu = rand_mask_loss.item()
            no_mask_loss_cpu = no_mask_loss.item()
            goal_mask_loss_cpu = goal_mask_loss.item()
            rand_mask_pose_loss_cpu = rand_mask_pose_loss.item()
            no_mask_pose_loss_cpu = no_mask_pose_loss.item()
            goal_mask_pose_loss_cpu = goal_mask_pose_loss.item()

            # Enhanced postfix with formatted numbers
            postfix_dict = {
                'rand': f"{format_number(loss_cpu, 3)}",
                'no_mask': f"{format_number(no_mask_loss_cpu, 3)}",
                'goal': f"{format_number(goal_mask_loss_cpu, 3)}",
                'pose': f"{format_number(rand_mask_pose_loss_cpu, 3)}"
            }
            tepoch.set_postfix(postfix_dict)

            # Use epoch*num_batches + i to ensure step increases continuously across epochs
            global_step = epoch * num_batches + i
            mlflow.log_metric("diffusion_eval_loss_random_masking", rand_mask_loss.item(), step=global_step)
            mlflow.log_metric("diffusion_eval_loss_no_masking", no_mask_loss.item(), step=global_step)
            mlflow.log_metric("diffusion_eval_loss_goal_masking", goal_mask_loss.item(), step=global_step)
            mlflow.log_metric("pose_eval_loss_random_masking", rand_mask_pose_loss.item(), step=global_step)
            mlflow.log_metric("pose_eval_loss_no_masking", no_mask_pose_loss.item(), step=global_step)
            mlflow.log_metric("pose_eval_loss_goal_masking", goal_mask_pose_loss.item(), step=global_step)

            if i % print_log_freq == 0 and print_log_freq != 0:
                losses = _compute_losses_nomad(
                    ema_model,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    distance.to(device),
                    actions.to(device),
                    device,
                    action_mask.to(device),
                    goal_pos.to(device),
                )

                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())

                # Use enhanced logger display instead of legacy logging
                loggers.display_metrics(epoch, i, num_batches, use_latest=True)

                data_log = {}
                for key, logger in loggers.all_loggers.items():
                    data_log[logger.full_name()] = logger.latest()

                if use_mlflow and i % mlflow_log_freq == 0 and mlflow_log_freq != 0:
                    for k, v in data_log.items():
                        # Additional safety check for NaN/infinite values before MLflow logging
                        if not np.isnan(v) and np.isfinite(v):
                            # Use epoch*num_batches + i to ensure step increases continuously across epochs
                            global_step = epoch * num_batches + i
                            mlflow.log_metric(k, v, step=global_step)
                        else:
                            print(f"Warning: Skipping MLflow logging for {k}={v} (NaN/infinite value)")

            if image_log_freq != 0 and i % image_log_freq == 0:
                handle_visualization(
                    using_prebuilt_dino and viz_dataloader is not None, viz_dataloader,
                    ema_model, noise_scheduler, batch_obs_images, batch_goal_images,
                    batch_viz_obs_images, batch_viz_goal_images, actions, distance,
                    goal_pos, device, eval_type, project_folder, epoch, num_images_log,
                    use_mlflow, i, num_batches, loggers, dataset_idx, mlflow_log_freq,
                    print_log_freq, image_log_freq, False
                )

            # Memory cleanup
            if i % 10 == 0:  # Every 10 batches
                cleanup_memory()


def train_eval_loop_nomad(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: DDPMScheduler,
    train_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    viz_dataloader: Optional[DataLoader],
    transform: transforms,
    goal_mask_prob: float,
    epochs: int,
    device: torch.device,
    project_folder: str,
    print_log_freq: int = 100,
    mlflow_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 1e-4,
    beta: float = 1e-4,
    use_mlflow: bool = True,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
):
    """
    Train and evaluate the model for several epochs (vint or gnm models)

    Args:
        model: model to train
        optimizer: optimizer to use
        lr_scheduler: learning rate scheduler to use
        noise_scheduler: noise scheduler to use
        dataloader: dataloader for train dataset
        test_dataloaders: dict of dataloaders for testing
        transform: transform to apply to images
        goal_mask_prob: probability of masking the goal token during training
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        mlflow_log_freq: frequency of logging to mlflow
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images to mlflow
        num_images_log: number of images to log to mlflow
        current_epoch: epoch to start training from
        alpha: weight of distance loss
        beta: weight of pose loss
        use_mlflow: whether to log to mlflow or not
        eval_fraction: fraction of training data to use for evaluation
        eval_freq: frequency of evaluation
    """
    latest_path = os.path.join(project_folder, f"latest.pth")
    ema_model = EMAModel(model=model, power=0.75)
    if viz_dataloader is not None:
        train_viz_dataloader, test_viz_dataloader = viz_dataloader
    else:
        train_viz_dataloader, test_viz_dataloader = None, None

    # Initialize training progress tracker
    total_epochs = current_epoch + epochs

    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            # Enhanced epoch start display
            print_section(
                f"Training Epoch {epoch + 1}/{total_epochs}",
                Symbols.TRAIN,
                Colors.BRIGHT_BLUE
            )
            epoch_start_time = time.time()
            train_nomad(
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                noise_scheduler=noise_scheduler,
                goal_mask_prob=goal_mask_prob,
                project_folder=project_folder,
                epoch=epoch,
                print_log_freq=print_log_freq,
                mlflow_log_freq=mlflow_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_mlflow=use_mlflow,
                alpha=alpha,
                beta=beta,
                viz_dataloader=train_viz_dataloader,
            )

            # Display training epoch completion
            if train_model:
                epoch_time = time.time() - epoch_start_time
                print_success(f"Training epoch completed in {format_time(epoch_time)}", Symbols.SUCCESS)



        # Enhanced model saving display
        print_info(f"Saving model checkpoints...", Symbols.SAVE)
        numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
        torch.save(ema_model.averaged_model.state_dict(), numbered_path)
        numbered_path = os.path.join(project_folder, f"ema_latest.pth")
        print_success(f"Saved EMA model to {numbered_path}", Symbols.SAVE)

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(model.state_dict(), numbered_path)
        torch.save(model.state_dict(), latest_path)
        print_success(f"Saved model to {numbered_path}", Symbols.SAVE)

        # save optimizer
        numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
        latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
        torch.save(optimizer.state_dict(), latest_optimizer_path)

        # save scheduler
        numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
        latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
        torch.save(lr_scheduler.state_dict(), latest_scheduler_path)

        if (epoch + 1) % eval_freq == 0:
            print_section(f"Evaluation Phase - Epoch {epoch + 1}", Symbols.EVAL, Colors.BRIGHT_MAGENTA)
            for dataset_type in test_dataloaders:
                print_info(f"Evaluating on {dataset_type.upper()} dataset", Symbols.TARGET)
                evaluate_nomad(
                    eval_type=dataset_type,
                    ema_model=ema_model,
                    dataloader=test_dataloaders[dataset_type],
                    transform=transform,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    goal_mask_prob=goal_mask_prob,
                    project_folder=project_folder,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    num_images_log=num_images_log,
                    mlflow_log_freq=mlflow_log_freq,
                    use_mlflow=use_mlflow,
                    eval_fraction=eval_fraction,
                    viz_dataloader=test_viz_dataloader,  # Pass visualization dataloader
                )
                
        if use_mlflow:
            # Use epoch as step for epoch-level metrics
            mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)

        if lr_scheduler is not None:
            lr_scheduler.step()


def load_checkpoint(model, model_type, checkpoint: dict) -> None:
    """Load model from checkpoint."""
    if model_type == "nomad":
        state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)


def load_ema_model(ema_model, state_dict: dict) -> None:
    """Load model from checkpoint."""
    ema_model.load_state_dict(state_dict)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    # print(table)
    print(f"Total Trainable Params: {total_params/1e6:.2f}M")
    return total_params


