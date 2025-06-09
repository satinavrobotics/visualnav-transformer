import itertools
import os
from typing import Optional

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
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import gc  # 🧹 Added for RAM/memory cleanup

from visualnav_transformer.train.vint_train.data.data_utils import (
    VISUALIZATION_IMAGE_SIZE,
)
from visualnav_transformer.train.vint_train.training.logger import Logger
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

# LOAD DATA CONFIG
from visualnav_transformer import ROOT_TRAIN
with open(
        # CHANGE
    # os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r"
    os.path.join(ROOT_TRAIN, "vint_train/data/data_config.yaml"), "r"
) as f:
    data_config = yaml.safe_load(f)
# POPULATE ACTION STATS
ACTION_STATS = {}
for key in data_config["action_stats"]:
    ACTION_STATS[key] = np.array(data_config["action_stats"][key])





# Train utils for ViNT and GNM
def _compute_losses(
    dist_label: torch.Tensor,
    action_label: torch.Tensor,
    dist_pred: torch.Tensor,
    action_pred: torch.Tensor,
    alpha: float,
    learn_angle: bool,
    action_mask: torch.Tensor = None,
):
    """
    Compute losses for distance and action prediction.

    """
    dist_loss = F.mse_loss(dist_pred.squeeze(-1), dist_label.float())

    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert (
            unreduced_loss.shape == action_mask.shape
        ), f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert (
        action_pred.shape == action_label.shape
    ), f"{action_pred.shape} != {action_label.shape}"
    action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none"))

    action_waypts_cos_similairity = action_reduce(
        F.cosine_similarity(action_pred[:, :, :2], action_label[:, :, :2], dim=-1)
    )
    multi_action_waypts_cos_sim = action_reduce(
        F.cosine_similarity(
            torch.flatten(action_pred[:, :, :2], start_dim=1),
            torch.flatten(action_label[:, :, :2], start_dim=1),
            dim=-1,
        )
    )

    results = {
        "dist_loss": dist_loss,
        "action_loss": action_loss,
        "action_waypts_cos_sim": action_waypts_cos_similairity,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
    }

    if learn_angle:
        action_orien_cos_sim = action_reduce(
            F.cosine_similarity(action_pred[:, :, 2:], action_label[:, :, 2:], dim=-1)
        )
        multi_action_orien_cos_sim = action_reduce(
            F.cosine_similarity(
                torch.flatten(action_pred[:, :, 2:], start_dim=1),
                torch.flatten(action_label[:, :, 2:], start_dim=1),
                dim=-1,
            )
        )
        results["action_orien_cos_sim"] = action_orien_cos_sim
        results["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim

    total_loss = alpha * 1e-2 * dist_loss + (1 - alpha) * action_loss
    results["total_loss"] = total_loss

    return results


def _log_data(
    i,
    epoch,
    num_batches,
    normalized,
    project_folder,
    num_images_log,
    loggers,
    obs_image,
    goal_image,
    action_pred,
    action_label,
    dist_pred,
    dist_label,
    goal_pos,
    dataset_index,
    use_mlflow,
    mode,
    use_latest,
    mlflow_log_freq=1,
    print_log_freq=1,
    image_log_freq=1,
    mlflow_increment_step=True,
):
    """
    Log data to mlflow and print to console.
    """
    data_log = {}
    for key, logger in loggers.items():
        if use_latest:
            data_log[logger.full_name()] = logger.latest()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(
                    f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}"
                )
        else:
            data_log[logger.full_name()] = logger.average()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) {logger.full_name()} {logger.average()}")

    if use_mlflow and i % mlflow_log_freq == 0 and mlflow_log_freq != 0:
        for k, v in data_log.items():
            # Use epoch*num_batches + i to ensure step increases continuously across epochs
            global_step = epoch * num_batches + i
            mlflow.log_metric(k, v, step=global_step if mlflow_increment_step else None)

    if image_log_freq != 0 and i % image_log_freq == 0:
        visualize_dist_pred(
            to_numpy(obs_image),
            to_numpy(goal_image),
            to_numpy(dist_pred),
            to_numpy(dist_label),
            mode,
            project_folder,
            epoch,
            num_images_log,
            use_mlflow=use_mlflow,
        )

        try:
            visualize_traj_pred(
                to_numpy(obs_image),
                to_numpy(goal_image),
                to_numpy(dataset_index),
                to_numpy(goal_pos),
                to_numpy(action_pred),
                to_numpy(action_label),
                mode,
                normalized,
                project_folder,
                epoch,
                num_images_log,
                use_mlflow=use_mlflow,
            )
        except Exception as e:
            print(f"\n[ERROR] Exception in visualize_traj_pred: {e}")
            print(f"  obs_image shape: {to_numpy(obs_image).shape}")
            print(f"  goal_image shape: {to_numpy(goal_image).shape}")
            print(f"  dataset_index shape: {to_numpy(dataset_index).shape}")
            print(f"  goal_pos shape: {to_numpy(goal_pos).shape}")
            print(f"  action_pred shape: {to_numpy(action_pred).shape}")
            print(f"  action_label shape: {to_numpy(action_label).shape}")


def train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int,
    alpha: float = 0.5,
    learn_angle: bool = True,
    print_log_freq: int = 100,
    mlflow_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_mlflow: bool = True,
    use_tqdm: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        learn_angle: whether to learn the angle of the action
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_mlflow: whether to use mlflow
        use_tqdm: whether to use tqdm
    """
    model.train()
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger(
        "action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    multi_action_waypts_cos_sim_logger = Logger(
        "multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    loggers = {
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "total_loss": total_loss_logger,
    }

    if learn_angle:
        action_orien_cos_sim_logger = Logger(
            "action_orien_cos_sim", "train", window_size=print_log_freq
        )
        multi_action_orien_cos_sim_logger = Logger(
            "multi_action_orien_cos_sim", "train", window_size=print_log_freq
        )
        loggers["action_orien_cos_sim"] = action_orien_cos_sim_logger
        loggers["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim_logger

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )
    for i, data in enumerate(tqdm_iter):
        (
            obs_image,
            goal_image,
            action_label,
            dist_label,
            goal_pos,
            dataset_index,
            action_mask,
        ) = data

        obs_images = torch.split(obs_image, 3, dim=1)
        viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)

        viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)

        goal_image = transform(goal_image).to(device)
        model_outputs = model(obs_image, goal_image)

        dist_label = dist_label.to(device)
        action_label = action_label.to(device)
        action_mask = action_mask.to(device)

        optimizer.zero_grad()

        dist_pred, action_pred = model_outputs

        losses = _compute_losses(
            dist_label=dist_label,
            action_label=action_label,
            dist_pred=dist_pred,
            action_pred=action_pred,
            alpha=alpha,
            learn_angle=learn_angle,
            action_mask=action_mask,
        )

        losses["total_loss"].backward()
        optimizer.step()

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        _log_data(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            normalized=normalized,
            project_folder=project_folder,
            num_images_log=num_images_log,
            loggers=loggers,
            obs_image=viz_obs_image,
            goal_image=viz_goal_image,
            action_pred=action_pred,
            action_label=action_label,
            dist_pred=dist_pred,
            dist_label=dist_label,
            goal_pos=goal_pos,
            dataset_index=dataset_index,
            mlflow_log_freq=mlflow_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
            use_mlflow=use_mlflow,
            mode="train",
            use_latest=True,
        )


def evaluate(
    eval_type: str,
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    num_images_log: int = 8,
    use_mlflow: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        model (nn.Module): model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        project_folder (string): path to project folder
        epoch (int): current epoch
        alpha (float): weight for action loss
        learn_angle (bool): whether to learn the angle of the action
        num_images_log (int): number of images to log
        use_mlflow (bool): whether to use mlflow for logging
        eval_fraction (float): fraction of data to use for evaluation
        use_tqdm (bool): whether to use tqdm for logging
    """
    model.eval()
    dist_loss_logger = Logger("dist_loss", eval_type)
    action_loss_logger = Logger("action_loss", eval_type)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", eval_type)
    multi_action_waypts_cos_sim_logger = Logger(
        "multi_action_waypts_cos_sim", eval_type
    )
    total_loss_logger = Logger("total_loss", eval_type)
    loggers = {
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "total_loss": total_loss_logger,
    }

    if learn_angle:
        action_orien_cos_sim_logger = Logger("action_orien_cos_sim", eval_type)
        multi_action_orien_cos_sim_logger = Logger(
            "multi_action_orien_cos_sim", eval_type
        )
        loggers["action_orien_cos_sim"] = action_orien_cos_sim_logger
        loggers["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim_logger

    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)

    viz_obs_image = None
    with torch.no_grad():
        tqdm_iter = tqdm.tqdm(
            itertools.islice(dataloader, num_batches),
            total=num_batches,
            disable=not use_tqdm,
            dynamic_ncols=True,
            desc=f"Evaluating {eval_type} for epoch {epoch}",
        )
        for i, data in enumerate(tqdm_iter):
            (
                obs_image,
                goal_image,
                action_label,
                dist_label,
                goal_pos,
                dataset_index,
                action_mask,
            ) = data

            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)

            goal_image = transform(goal_image).to(device)
            model_outputs = model(obs_image, goal_image)

            dist_label = dist_label.to(device)
            action_label = action_label.to(device)
            action_mask = action_mask.to(device)

            dist_pred, action_pred = model_outputs

            losses = _compute_losses(
                dist_label=dist_label,
                action_label=action_label,
                dist_pred=dist_pred,
                action_pred=action_pred,
                alpha=alpha,
                learn_angle=learn_angle,
                action_mask=action_mask,
            )

            for key, value in losses.items():
                if key in loggers:
                    logger = loggers[key]
                    logger.log_data(value.item())

    # Log data to mlflow/console, with visualizations selected from the last batch
    _log_data(
        i=i,
        epoch=epoch,
        num_batches=num_batches,
        normalized=normalized,
        project_folder=project_folder,
        num_images_log=num_images_log,
        loggers=loggers,
        obs_image=viz_obs_image,
        goal_image=viz_goal_image,
        action_pred=action_pred,
        action_label=action_label,
        goal_pos=goal_pos,
        dist_pred=dist_pred,
        dist_label=dist_label,
        dataset_index=dataset_index,
        use_mlflow=use_mlflow,
        mode=eval_type,
        use_latest=False,
        mlflow_increment_step=False,
    )

    return (
        dist_loss_logger.average(),
        action_loss_logger.average(),
        total_loss_logger.average(),
    )


# Train utils for NOMAD


def _compute_losses_nomad(
    ema_model,
    noise_scheduler,
    batch_obs_images,
    batch_goal_images,
    batch_dist_label: torch.Tensor,
    batch_action_label: torch.Tensor,
    device: torch.device,
    action_mask: torch.Tensor,
):
    """
    Compute losses for distance and action prediction.
    """

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    model_output_dict = model_output(
        ema_model,
        noise_scheduler,
        batch_obs_images,
        batch_goal_images,
        pred_horizon,
        action_dim,
        num_samples=1,
        device=device,
    )
    uc_actions = model_output_dict["uc_actions"]
    gc_actions = model_output_dict["gc_actions"]
    gc_distance = model_output_dict["gc_distance"]

    gc_dist_loss = F.mse_loss(gc_distance, batch_dist_label.unsqueeze(-1))

    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert (
            unreduced_loss.shape == action_mask.shape
        ), f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert (
        uc_actions.shape == batch_action_label.shape
    ), f"{uc_actions.shape} != {batch_action_label.shape}"
    assert (
        gc_actions.shape == batch_action_label.shape
    ), f"{gc_actions.shape} != {batch_action_label.shape}"

    uc_action_loss = action_reduce(
        F.mse_loss(uc_actions, batch_action_label, reduction="none")
    )
    gc_action_loss = action_reduce(
        F.mse_loss(gc_actions, batch_action_label, reduction="none")
    )

    uc_action_waypts_cos_similairity = action_reduce(
        F.cosine_similarity(uc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1)
    )
    uc_multi_action_waypts_cos_sim = action_reduce(
        F.cosine_similarity(
            torch.flatten(uc_actions[:, :, :2], start_dim=1),
            torch.flatten(batch_action_label[:, :, :2], start_dim=1),
            dim=-1,
        )
    )

    gc_action_waypts_cos_similairity = action_reduce(
        F.cosine_similarity(gc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1)
    )
    gc_multi_action_waypts_cos_sim = action_reduce(
        F.cosine_similarity(
            torch.flatten(gc_actions[:, :, :2], start_dim=1),
            torch.flatten(batch_action_label[:, :, :2], start_dim=1),
            dim=-1,
        )
    )

    results = {
        "uc_action_loss": uc_action_loss,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_similairity,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim,
        "gc_dist_loss": gc_dist_loss,
        "gc_action_loss": gc_action_loss,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_similairity,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim,
    }

    return results


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
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_mlflow: whether to use mlflow
    """
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    num_batches = len(dataloader)

    uc_action_loss_logger = Logger(
        "uc_action_loss", "train", window_size=print_log_freq
    )
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", "train", window_size=print_log_freq)
    gc_action_loss_logger = Logger(
        "gc_action_loss", "train", window_size=print_log_freq
    )
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    # Check if using pre-built DINO features
    # For ConcatDataset, we need to check the first dataset
    if hasattr(dataloader.dataset, 'datasets'):
        # This is a ConcatDataset
        first_dataset = dataloader.dataset.datasets[0]
        using_prebuilt_dino = hasattr(first_dataset, 'prebuilt_dino') and first_dataset.prebuilt_dino
    else:
        # This is a regular dataset
        using_prebuilt_dino = hasattr(dataloader.dataset, 'prebuilt_dino') and dataloader.dataset.prebuilt_dino

    if using_prebuilt_dino:
        print("Using pre-built DINO features for training")

    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, # obs_image shape depends on whether using pre-built DINO features, else: goal_image shape: torch.Size([128, 3, 240, 320]),
                goal_image, # goal_image shape depends on whether using pre-built DINO features, else: goal_image shape: torch.Size([128, 3, 240, 320]),
                actions, # actions shape: torch.Size([128, 8, 2]),
                distance, # distance shape: torch.Size([128]),
                goal_pos, # goal_pos shape: torch.Size([128, 2]),
                dataset_idx, # dataset_idx shape: torch.Size([128]),
                action_mask, # action_mask shape: torch.Size([128])
            ) = data

            # Handle differently based on whether using pre-built DINO features
            if using_prebuilt_dino:
                # With pre-built DINO features, obs_image and goal_image are already feature vectors
                # No need to split, resize, or transform
                batch_obs_images = obs_image.to(device)  # These are already DINO features
                batch_goal_images = goal_image.to(device)  # These are already DINO features

                # Set visualization images to None, will be fetched during visualization if needed
                batch_viz_obs_images = None
                batch_viz_goal_images = None
            else:
                # Regular processing for raw images
                obs_images = torch.split(obs_image, 3, dim=1)
                batch_viz_obs_images = TF.resize(
                    obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1]
                )
                batch_viz_goal_images = TF.resize(
                    goal_image, VISUALIZATION_IMAGE_SIZE[::-1]
                )
                batch_obs_images = [transform(obs) for obs in obs_images]
                batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
                batch_goal_images = transform(goal_image).to(device)

            action_mask = action_mask.to(device)

            B = actions.shape[0]

            # Generate random goal mask
            goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            obsgoal_cond = model(
                "vision_encoder",
                obs_img=batch_obs_images,
                goal_img=batch_goal_images,
                input_goal_mask=goal_mask,
            )

            # Get distance label
            distance = distance.float().to(device)

            deltas = get_delta(actions)
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(ndeltas).to(device)
            assert naction.shape[-1] == 2, "action dim must be 2"

            # Predict distance
            dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance)
            dist_loss = (dist_loss * (1 - goal_mask.float())).mean() / (
                1e-2 + (1 - goal_mask.float()).mean()
            )

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

            def action_reduce(unreduced_loss: torch.Tensor):
                # Reduce over non-batch dimensions to get loss per batch element
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                assert (
                    unreduced_loss.shape == action_mask.shape
                ), f"{unreduced_loss.shape} != {action_mask.shape}"
                return (unreduced_loss * action_mask).mean() / (
                    action_mask.mean() + 1e-2
                )

            # L2 loss
            diffusion_loss = action_reduce(
                F.mse_loss(noise_pred, noise, reduction="none")
            )

            # Total loss
            loss = alpha * dist_loss + (1 - alpha) * diffusion_loss

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Exponential Moving Average of the model weights
            ema_model.step(model)

            # Logging
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            # Use epoch*num_batches + i to ensure step increases continuously across epochs
            global_step = epoch * num_batches + i
            mlflow.log_metric("total_loss", loss_cpu, step=global_step)
            mlflow.log_metric("dist_loss", dist_loss.item(), step=global_step)
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
                )

                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())

                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(
                            f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}"
                        )

                if use_mlflow and i % mlflow_log_freq == 0 and mlflow_log_freq != 0:
                    for k, v in data_log.items():
                        # Use epoch*num_batches + i to ensure step increases continuously across epochs
                        global_step = epoch * num_batches + i
                        mlflow.log_metric(k, v, step=global_step)


            if image_log_freq != 0 and i % image_log_freq == 0:
                # When using pre-built DINO features, we need to use the visualization dataset
                if using_prebuilt_dino and viz_dataloader is not None:
                    try:
                        # Get visualization batch
                        viz_batch = next(iter(viz_dataloader))

                        # Unpack the visualization batch
                        viz_obs_img, viz_goal_img, viz_actions, viz_distance, viz_goal_pos, viz_dataset_idx, viz_action_mask = viz_batch

                        # Move to device
                        viz_obs_img = viz_obs_img.to(device)
                        viz_goal_img = viz_goal_img.to(device)
                        viz_goal_pos = viz_goal_pos.to(device)

                        # Extract visualization images (for display only)
                        obs_images = torch.split(viz_obs_img, 3, dim=1)
                        viz_obs_images = obs_images[-1]

                        # Call visualization function with visualization images
                        visualize_diffusion_action_distribution(
                            ema_model.averaged_model,
                            noise_scheduler,
                            batch_obs_images,
                            batch_goal_images,
                            viz_obs_images,  # Use visualization images
                            viz_goal_img,  # Use visualization images
                            actions,  # This is batch_action_label
                            distance,  # This is batch_distance_labels
                            viz_goal_pos,  # Use visualization goal positions
                            device,
                            "train",
                            project_folder,
                            epoch,
                            min(num_images_log, len(viz_obs_images)),
                            30,
                            use_mlflow,
                            # Additional parameters needed for _log_data
                            i=i,
                            num_batches=num_batches,
                            loggers=loggers,
                            dataset_index=viz_dataset_idx,
                            mlflow_log_freq=mlflow_log_freq,
                            print_log_freq=print_log_freq,
                            image_log_freq=image_log_freq,
                            use_latest=True,
                        )
                    except Exception as e:
                        print(f"Error during visualization: {e}")
                        # Continue training even if visualization fails
                else:
                    # Original visualization code
                    visualize_diffusion_action_distribution(
                        ema_model.averaged_model,
                        noise_scheduler,
                        batch_obs_images,
                        batch_goal_images,
                        batch_viz_obs_images,
                        batch_viz_goal_images,
                        actions,  # This is batch_action_label
                        distance,  # This is batch_distance_labels
                        goal_pos,
                        device,
                        "train",
                        project_folder,
                        epoch,
                        num_images_log,
                        30,
                        use_mlflow,
                        # Additional parameters needed for _log_data
                        i=i,
                        num_batches=num_batches,
                        loggers=loggers,
                        dataset_index=dataset_idx,
                        mlflow_log_freq=mlflow_log_freq,
                        print_log_freq=print_log_freq,
                        image_log_freq=image_log_freq,
                        use_latest=True,
                    )


             # === 🧹 MEMORY CLEANUP AFTER BATCH ===
            # Create a list of variables to delete
            vars_to_delete = [
                obs_image, goal_image, actions, distance, goal_pos, dataset_idx,
                batch_obs_images, batch_goal_images,
                obsgoal_cond, dist_pred, noise, timesteps, noisy_action,
                noise_pred, naction, deltas, ndeltas
            ]

            # Add optional variables if they exist
            if not using_prebuilt_dino:
                vars_to_delete.append(obs_images)

            if batch_viz_obs_images is not None:
                vars_to_delete.append(batch_viz_obs_images)

            if batch_viz_goal_images is not None:
                vars_to_delete.append(batch_viz_goal_images)

            # Delete all variables
            for var in vars_to_delete:
                del var

            # More aggressive memory cleanup
            torch.cuda.empty_cache()  # 🧹 clear cached GPU memory
            gc.collect()              # 🧹 run Python garbage collector

            # Add a small sleep to allow memory to be properly freed
            if i % 10 == 0:  # Every 10 batches, do a more thorough cleanup
                import time
                time.sleep(0.1)  # Small sleep to allow memory to be freed


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

    uc_action_loss_logger = Logger(
        "uc_action_loss", eval_type, window_size=print_log_freq
    )
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", eval_type, window_size=print_log_freq)
    gc_action_loss_logger = Logger(
        "gc_action_loss", eval_type, window_size=print_log_freq
    )
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    num_batches = max(int(num_batches * eval_fraction), 1)

    # Check if using pre-built DINO features
    # For ConcatDataset, we need to check the first dataset
    if hasattr(dataloader.dataset, 'datasets'):
        # This is a ConcatDataset
        first_dataset = dataloader.dataset.datasets[0]
        using_prebuilt_dino = hasattr(first_dataset, 'prebuilt_dino') and first_dataset.prebuilt_dino
    else:
        # This is a regular dataset
        using_prebuilt_dino = hasattr(dataloader.dataset, 'prebuilt_dino') and dataloader.dataset.prebuilt_dino

    if using_prebuilt_dino:
        print(f"Using pre-built DINO features for evaluation ({eval_type})")

    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        dynamic_ncols=True,
        desc=f"Evaluating {eval_type} for epoch {epoch}",
        leave=False,
    ) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image,
                goal_image,
                actions,
                distance,
                goal_pos,
                dataset_idx,
                action_mask,
            ) = data

            # Handle differently based on whether using pre-built DINO features
            if using_prebuilt_dino:
                # With pre-built DINO features, obs_image and goal_image are already feature vectors
                # No need to split, resize, or transform
                batch_obs_images = obs_image.to(device)  # These are already DINO features
                batch_goal_images = goal_image.to(device)  # These are already DINO features

                # Set visualization images to None, will be fetched during visualization if needed
                batch_viz_obs_images = None
                batch_viz_goal_images = None
            else:
                # Regular processing for raw images
                obs_images = torch.split(obs_image, 3, dim=1)
                batch_viz_obs_images = TF.resize(
                    obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1]
                )
                batch_viz_goal_images = TF.resize(
                    goal_image, VISUALIZATION_IMAGE_SIZE[::-1]
                )
                batch_obs_images = [transform(obs) for obs in obs_images]
                batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
                batch_goal_images = transform(goal_image).to(device)

            action_mask = action_mask.to(device)

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

            distance = distance.to(device)

            deltas = get_delta(actions)
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(ndeltas).to(device)
            assert naction.shape[-1] == 2, "action dim must be 2"

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

            # Logging
            loss_cpu = rand_mask_loss.item()
            tepoch.set_postfix(loss=loss_cpu)

            # Use epoch*num_batches + i to ensure step increases continuously across epochs
            global_step = epoch * num_batches + i
            mlflow.log_metric("diffusion_eval_loss_random_masking", rand_mask_loss.item(), step=global_step)
            mlflow.log_metric("diffusion_eval_loss_no_masking", no_mask_loss.item(), step=global_step)
            mlflow.log_metric("diffusion_eval_loss_goal_masking", goal_mask_loss.item(), step=global_step)

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
                )

                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())

                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(
                            f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}"
                        )

                if use_mlflow and i % mlflow_log_freq == 0 and mlflow_log_freq != 0:
                    for k, v in data_log.items():
                        # Use epoch*num_batches + i to ensure step increases continuously across epochs
                        global_step = epoch * num_batches + i
                        mlflow.log_metric(k, v, step=global_step)

            if image_log_freq != 0 and i % image_log_freq == 0:
                # When using pre-built DINO features, we need to use the visualization dataset
                if using_prebuilt_dino and viz_dataloader is not None:
                    try:
                        # Get visualization batch
                        viz_batch = next(iter(viz_dataloader))

                        # Unpack the visualization batch
                        viz_obs_img, viz_goal_img, viz_actions, viz_distance, viz_goal_pos, viz_dataset_idx, viz_action_mask = viz_batch

                        # Move to device
                        viz_obs_img = viz_obs_img.to(device)
                        viz_goal_img = viz_goal_img.to(device)
                        viz_goal_pos = viz_goal_pos.to(device)

                        # Extract visualization images (for display only)
                        obs_images = torch.split(viz_obs_img, 3, dim=1)
                        viz_obs_images = obs_images[-1]

                        # Call visualization function with visualization images
                        visualize_diffusion_action_distribution(
                            ema_model,
                            noise_scheduler,
                            batch_obs_images,
                            batch_goal_images,
                            viz_obs_images,  # Use visualization images
                            viz_goal_img,  # Use visualization images
                            actions,  # This is batch_action_label
                            distance,  # This is batch_distance_labels
                            viz_goal_pos,  # Use visualization goal positions
                            device,
                            eval_type,
                            project_folder,
                            epoch,
                            min(num_images_log, len(viz_obs_images)),
                            30,
                            use_mlflow,
                            # Additional parameters needed for _log_data
                            i=i,
                            num_batches=num_batches,
                            loggers=loggers,
                            dataset_index=viz_dataset_idx,
                            mlflow_log_freq=mlflow_log_freq,
                            print_log_freq=print_log_freq,
                            image_log_freq=image_log_freq,
                            use_latest=False,
                        )
                    except Exception as e:
                        print(f"Error during visualization: {e}")
                        # Continue evaluation even if visualization fails
                else:
                    # Original visualization code
                    visualize_diffusion_action_distribution(
                        ema_model,
                        noise_scheduler,
                        batch_obs_images,
                        batch_goal_images,
                        batch_viz_obs_images,
                        batch_viz_goal_images,
                        actions,  # This is batch_action_label
                        distance,  # This is batch_distance_labels
                        goal_pos,
                        device,
                        eval_type,
                        project_folder,
                        epoch,
                        num_images_log,
                        30,
                        use_mlflow,
                        # Additional parameters needed for _log_data
                        i=i,
                        num_batches=num_batches,
                        loggers=loggers,
                        dataset_index=dataset_idx,
                        mlflow_log_freq=mlflow_log_freq,
                        print_log_freq=print_log_freq,
                        image_log_freq=image_log_freq,
                        use_latest=False,
                    )

                # === 🧹 MEMORY CLEANUP AFTER BATCH ===
                # Create a list of variables to delete
                vars_to_delete = [
                    obs_image, goal_image, actions, distance, goal_pos, dataset_idx,
                    batch_obs_images, batch_goal_images, rand_goal_mask, goal_mask, no_mask,
                    rand_mask_cond, obsgoal_cond, goal_mask_cond, noise, timesteps, noisy_actions,
                    rand_mask_noise_pred, no_mask_noise_pred, goal_mask_noise_pred
                ]

                # Add optional variables if they exist
                if not using_prebuilt_dino and 'obs_images' in locals():
                    vars_to_delete.append(obs_images)

                if 'batch_viz_obs_images' in locals() and batch_viz_obs_images is not None:
                    vars_to_delete.append(batch_viz_obs_images)

                if 'batch_viz_goal_images' in locals() and batch_viz_goal_images is not None:
                    vars_to_delete.append(batch_viz_goal_images)

                # Delete all variables
                for var in vars_to_delete:
                    try:
                        del var
                    except:
                        pass  # Ignore if variable doesn't exist

                # More aggressive memory cleanup
                torch.cuda.empty_cache()  # 🧹 clear cached GPU memory
                gc.collect()              # 🧹 run Python garbage collector

                # Add a small sleep to allow memory to be properly freed
                if i % 10 == 0:  # Every 10 batches, do a more thorough cleanup
                    import time
                    time.sleep(0.1)  # Small sleep to allow memory to be freed


# normalize data
def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    return stats


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data


def get_delta(actions):
    # append zeros to first action
    ex_actions = np.concatenate(
        [np.zeros((actions.shape[0], 1, actions.shape[-1])), actions], axis=1
    )
    delta = ex_actions[:, 1:] - ex_actions[:, :-1]
    return delta


def get_action(diffusion_output, action_stats=ACTION_STATS):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)


def model_output(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
):
    goal_mask = torch.ones((batch_goal_images.shape[0],)).long().to(device)
    obs_cond = model(
        "vision_encoder",
        obs_img=batch_obs_images,
        goal_img=batch_goal_images,
        input_goal_mask=goal_mask,
    )
    # obs_cond = obs_cond.flatten(start_dim=1)
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)

    no_mask = torch.zeros((batch_goal_images.shape[0],)).long().to(device)
    obsgoal_cond = model(
        "vision_encoder",
        obs_img=batch_obs_images,
        goal_img=batch_goal_images,
        input_goal_mask=no_mask,
    )
    # obsgoal_cond = obsgoal_cond.flatten(start_dim=1)
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device
    )
    diffusion_output = noisy_diffusion_output

    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obs_cond,
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred, timestep=k, sample=diffusion_output
        ).prev_sample

    uc_actions = get_action(diffusion_output, ACTION_STATS)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device
    )
    diffusion_output = noisy_diffusion_output

    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obsgoal_cond,
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred, timestep=k, sample=diffusion_output
        ).prev_sample
    obsgoal_cond = obsgoal_cond.flatten(start_dim=1)
    gc_actions = get_action(diffusion_output, ACTION_STATS)
    gc_distance = model("dist_pred_net", obsgoal_cond=obsgoal_cond)

    return {
        "uc_actions": uc_actions,
        "gc_actions": gc_actions,
        "gc_distance": gc_distance,
    }


def visualize_diffusion_action_distribution(
    ema_model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    batch_viz_obs_images: torch.Tensor,
    batch_viz_goal_images: torch.Tensor,
    batch_action_label: torch.Tensor,  # This is 'actions' from train_nomad
    batch_distance_labels: torch.Tensor,  # This is 'distance' from train_nomad
    batch_goal_pos: torch.Tensor,
    device: torch.device,
    eval_type: str,
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,
    use_mlflow: bool = True,
    # Additional parameters needed for _log_data
    i: int = 0,
    num_batches: int = 1,
    loggers: dict = None,
    dataset_index: torch.Tensor = None,
    mlflow_log_freq: int = 1,
    print_log_freq: int = 1,
    image_log_freq: int = 1,
    use_latest: bool = True,
):
    """
    Plot samples from the exploration model.

    Note: When using pre-built DINO features, batch_viz_obs_images and batch_viz_goal_images
    may be None. In this case, we'll skip the visualization of the images.
    """

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    max_batch_size = batch_obs_images.shape[0]

    num_images_log = min(
        num_images_log,
        batch_obs_images.shape[0],
        batch_goal_images.shape[0],
        batch_action_label.shape[0],
        batch_goal_pos.shape[0],
    )
    batch_obs_images = batch_obs_images[:num_images_log]
    batch_goal_images = batch_goal_images[:num_images_log]
    batch_action_label = batch_action_label[:num_images_log]
    batch_goal_pos = batch_goal_pos[:num_images_log]

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    # split into batches
    batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
    batch_goal_images_list = torch.split(batch_goal_images, max_batch_size, dim=0)

    uc_actions_list = []
    gc_actions_list = []
    gc_distances_list = []

    for obs, goal in zip(batch_obs_images_list, batch_goal_images_list):
        model_output_dict = model_output(
            ema_model,
            noise_scheduler,
            obs,
            goal,
            pred_horizon,
            action_dim,
            num_samples,
            device,
        )
        uc_actions_list.append(to_numpy(model_output_dict["uc_actions"]))
        gc_actions_list.append(to_numpy(model_output_dict["gc_actions"]))
        gc_distances_list.append(to_numpy(model_output_dict["gc_distance"]))

    # concatenate
    uc_actions_list = np.concatenate(uc_actions_list, axis=0)
    gc_actions_list = np.concatenate(gc_actions_list, axis=0)
    gc_distances_list = np.concatenate(gc_distances_list, axis=0)

    # split into actions per observation
    uc_actions_list = np.split(uc_actions_list, num_images_log, axis=0)
    gc_actions_list = np.split(gc_actions_list, num_images_log, axis=0)
    gc_distances_list = np.split(gc_distances_list, num_images_log, axis=0)

    gc_distances_avg = [np.mean(dist) for dist in gc_distances_list]
    gc_distances_std = [np.std(dist) for dist in gc_distances_list]

    assert len(uc_actions_list) == len(gc_actions_list) == num_images_log

    np_distance_labels = to_numpy(batch_distance_labels)

    for img_idx in range(num_images_log):
        fig, ax = plt.subplots(1, 3)
        uc_actions = uc_actions_list[img_idx]
        gc_actions = gc_actions_list[img_idx]
        action_label = to_numpy(batch_action_label[img_idx])

        traj_list = np.concatenate(
            [
                uc_actions,
                gc_actions,
                action_label[None],
            ],
            axis=0,
        )

        # traj_labels = ["r", "GC", "GC_mean", "GT"]
        traj_colors = (
            ["red"] * len(uc_actions) + ["green"] * len(gc_actions) + ["yellow"]
        )
        traj_alphas = [0.1] * (len(uc_actions) + len(gc_actions)) + [1.0]

        # make points numpy array of robot positions (0, 0) and goal positions
        point_list = [np.array([0, 0]), to_numpy(batch_goal_pos[img_idx])]
        point_colors = ["green", "red"]
        point_alphas = [1.0, 1.0]

        try:
            plot_trajs_and_points(
                ax[0],
                traj_list,
                point_list,
                traj_colors,
                point_colors,
                traj_labels=None,
                point_labels=None,
                quiver_freq=0,
                traj_alphas=traj_alphas,
                point_alphas=point_alphas,
            )
        except Exception as e:
            print(f"\n[ERROR] Exception in plot_trajs_and_points: {e}")
            print(f"  traj_list shape: {traj_list.shape if hasattr(traj_list, 'shape') else 'not array'}")
            print(f"  point_list shapes: {[point.shape if hasattr(point, 'shape') else 'not array' for point in point_list]}")
            raise

        # Extract model outputs for _log_data
        # Use the goal-conditioned actions as predictions
        action_pred = from_numpy(gc_actions).to(device)
        # Use the average distance prediction
        dist_pred = from_numpy(np.array([gc_distances_avg[img_idx]])).to(device).unsqueeze(-1)

        # If dataset_index is not provided, create a dummy one
        local_dataset_index = dataset_index[img_idx:img_idx+1] if dataset_index is not None else torch.zeros(1, dtype=torch.long).to(device)

        # Use img_idx instead of i to index the batch tensors
        if batch_viz_obs_images is not None and batch_viz_goal_images is not None:
            obs_image = to_numpy(batch_viz_obs_images[img_idx])
            goal_image = to_numpy(batch_viz_goal_images[img_idx])
            # move channel to last dimension
            obs_image = np.moveaxis(obs_image, 0, -1)
            goal_image = np.moveaxis(goal_image, 0, -1)
            ax[1].imshow(obs_image)
            ax[2].imshow(goal_image)
        else:
            # If visualization images are not available, show placeholder
            ax[1].text(0.5, 0.5, "No observation image\n(using pre-built DINO features)",
                      horizontalalignment='center', verticalalignment='center')
            ax[2].text(0.5, 0.5, "No goal image\n(using pre-built DINO features)",
                      horizontalalignment='center', verticalalignment='center')
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            ax[2].set_xticks([])
            ax[2].set_yticks([])

        # set title
        ax[0].set_title(f"diffusion action predictions")
        ax[1].set_title(f"observation")
        ax[2].set_title(
            f"goal: label={np_distance_labels[img_idx]} gc_dist={gc_distances_avg[img_idx]:.2f}±{gc_distances_std[img_idx]:.2f}"
        )

        # make the plot large
        fig.set_size_inches(18.5, 10.5)

        save_path = os.path.join(visualize_path, f"sample_{img_idx}.png")
        plt.savefig(save_path)
        if use_mlflow:
            mlflow.log_artifact(save_path, artifact_path=f"{eval_type}/action_samples")
        plt.close(fig)

        # Call _log_data with all required parameters
        if loggers is not None:
            # Skip image logging if visualization images are not available
            if batch_viz_obs_images is not None and batch_viz_goal_images is not None:
                _log_data(
                    i=i,  # Use the original batch index
                    epoch=epoch,
                    num_batches=num_batches,
                    normalized=False,  # Always set to False we already normalize
                    project_folder=project_folder,
                    num_images_log=num_images_log,
                    loggers=loggers,
                    obs_image=batch_viz_obs_images[img_idx:img_idx+1],  # Use only the current image
                    goal_image=batch_viz_goal_images[img_idx:img_idx+1],  # Use only the current image
                    action_pred=action_pred,  # Use the goal-conditioned actions as predictions
                    action_label=batch_action_label[img_idx:img_idx+1],  # This is 'actions' from train_nomad
                    dist_pred=dist_pred,  # Use the average distance prediction
                    dist_label=batch_distance_labels[img_idx:img_idx+1],  # This is 'distance' from train_nomad
                    goal_pos=batch_goal_pos[img_idx:img_idx+1],  # Use only the current goal position
                    dataset_index=local_dataset_index,
                    mlflow_log_freq=mlflow_log_freq,
                    print_log_freq=print_log_freq,
                    image_log_freq=image_log_freq,
                    use_mlflow=use_mlflow,
                    mode=eval_type,  # Use the provided eval_type instead of hardcoding "train"
                    use_latest=use_latest,
                )
            else:
                # Log metrics without images
                for logger_name, logger in loggers.items():
                    if logger_name == "action_loss":
                        # Log action prediction error
                        action_error = torch.mean(torch.abs(action_pred - batch_action_label[img_idx:img_idx+1]))
                        logger.log_data(action_error.item())
                    elif logger_name == "dist_loss":
                        # Log distance prediction error
                        dist_error = torch.mean(torch.abs(dist_pred - batch_distance_labels[img_idx:img_idx+1].unsqueeze(-1)))
                        logger.log_data(dist_error.item())
                    elif logger_name == "total_loss":
                        # Log total loss (weighted sum of action and distance losses)
                        action_error = torch.mean(torch.abs(action_pred - batch_action_label[img_idx:img_idx+1]))
                        dist_error = torch.mean(torch.abs(dist_pred - batch_distance_labels[img_idx:img_idx+1].unsqueeze(-1)))
                        total_error = action_error + dist_error
                        logger.log_data(total_error.item())

        del action_pred, dist_pred, local_dataset_index
        torch.cuda.empty_cache()