import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import mlflow
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from prettytable import PrettyTable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from visualnav_transformer.train.vint_train.training.train_utils import (
    evaluate,
    evaluate_nomad,
    train,
    train_nomad,
)


def train_eval_loop(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    dataloader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    epochs: int,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    mlflow_log_freq: int = 10,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    use_mlflow: bool = True,
    eval_fraction: float = 0.25,
):
    """
    Train and evaluate the model for several epochs (vint or gnm models)

    Args:
        train_model: whether to train the model or not
        model: model to train
        optimizer: optimizer to use
        scheduler: learning rate scheduler to use
        dataloader: dataloader for train dataset
        test_dataloaders: dict of dataloaders for testing
        transform: transform to apply to images
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        normalized: whether to normalize the action space or not
        mlflow_log_freq: frequency of logging to mlflow
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images to mlflow
        num_images_log: number of images to log to mlflow
        current_epoch: epoch to start training from
        alpha: tradeoff between distance and action loss
        learn_angle: whether to learn the angle or not
        use_mlflow: whether to log to mlflow or not
        eval_fraction: fraction of training data to use for evaluation
    """
    assert 0 <= alpha <= 1
    latest_path = os.path.join(project_folder, f"latest.pth")

    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            print(f"Start ViNT Training Epoch {epoch}/{current_epoch + epochs - 1}")
            train(
                model=model,
                optimizer=optimizer,
                dataloader=dataloader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                print_log_freq=print_log_freq,
                mlflow_log_freq=mlflow_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_mlflow=use_mlflow,
            )

        avg_total_test_loss = []
        for dataset_type in test_dataloaders:
            print(
                f"Start {dataset_type} ViNT Testing Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            loader = test_dataloaders[dataset_type]

            test_dist_loss, test_action_loss, total_eval_loss = evaluate(
                eval_type=dataset_type,
                model=model,
                dataloader=loader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                num_images_log=num_images_log,
                use_mlflow=use_mlflow,
                eval_fraction=eval_fraction,
            )

            avg_total_test_loss.append(total_eval_loss)

        if use_mlflow:
            # Use epoch as step for epoch-level metrics
            mlflow.log_metric("avg_total_test_loss", np.mean(avg_total_test_loss), step=epoch)
            mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)

        checkpoint = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "avg_total_test_loss": np.mean(avg_total_test_loss),
            "scheduler": scheduler,
        }

        if scheduler is not None:
            # scheduler calls based on the type of scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(np.mean(avg_total_test_loss))
            else:
                scheduler.step()

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(checkpoint, latest_path)
        torch.save(checkpoint, numbered_path)  # keep track of model at every epoch

    print()


def train_eval_loop_nomad(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: DDPMScheduler,
    train_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
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
        alpha: tradeoff between distance and action loss
        use_mlflow: whether to log to mlflow or not
        eval_fraction: fraction of training data to use for evaluation
        eval_freq: frequency of evaluation
    """
    latest_path = os.path.join(project_folder, f"latest.pth")
    ema_model = EMAModel(model=model, power=0.75)

    # Check if using pre-built DINO features
    if hasattr(train_loader.dataset, 'datasets'):
        # This is a ConcatDataset, get attributes from the first dataset
        first_dataset = train_loader.dataset.datasets[0]
        using_prebuilt_dino = hasattr(first_dataset, 'prebuilt_dino') and first_dataset.prebuilt_dino
    else:
        # Regular dataset
        using_prebuilt_dino = hasattr(train_loader.dataset, 'prebuilt_dino') and train_loader.dataset.prebuilt_dino

    # Only load visualization datasets if using pre-built DINO features
    train_viz_dataloader = None
    test_viz_dataloaders = {}

    if using_prebuilt_dino:
        print("Loading visualization datasets for pre-built DINO features")

        # For visualization, we'll use a small subset of the regular datasets
        # We'll use the first available dataset for visualization
        from visualnav_transformer.train.vint_train.data.vint_dataset import ViNT_Dataset
        from visualnav_transformer import ROOT_TRAIN
        import yaml

        # Load data config
        with open(os.path.join(ROOT_TRAIN, "vint_train/data/data_config.yaml"), "r") as f:
            data_config = yaml.safe_load(f)

        # Use the dedicated visualization dataset
        viz_dataset_name = "viz_data"
        viz_dataset_config = data_config["datasets"].get(viz_dataset_name)

        if viz_dataset_config is None or not viz_dataset_config.get("available", False):
            print("Warning: Visualization dataset not found or not available")

            # Fall back to the first available dataset if viz_data is not available
            for dataset_name, config in data_config["datasets"].items():
                if config.get("available", False) and config.get("split", 0.0) > 0:
                    viz_dataset_name = dataset_name
                    viz_dataset_config = config
                    break

            if viz_dataset_name is None:
                print("Warning: No available dataset found for visualization")
            else:
                print(f"Falling back to {viz_dataset_name} for visualization")
        else:
            print(f"Using dedicated visualization dataset: {viz_dataset_name}")

            # Get dataset parameters from the first dataset in the train_loader
            if hasattr(train_loader.dataset, 'datasets'):
                first_dataset = train_loader.dataset.datasets[0]
            else:
                first_dataset = train_loader.dataset

            # Create a small visualization dataset using ViNT_Dataset (not FeatureDataset)
            # The viz_data contains raw images with LMDB cache, not pre-built DINO features

            # Create train visualization dataset - use a small subset
            train_viz_folder = os.path.join(viz_dataset_config["data_folder"], "train_viz")
            print(f"Loading train visualization dataset from: {train_viz_folder}")

            try:
                train_viz_dataset = ViNT_Dataset(
                    data_folder=train_viz_folder,
                    split="train",
                    split_ratio=1.0,  # Use all available trajectories in train_viz
                    dataset_name=f"{viz_dataset_name}_train_viz",  # Unique name for viz dataset
                    dataset_index=0,  # Use 0 for visualization
                    image_size=first_dataset.image_size,
                    waypoint_spacing=viz_dataset_config.get("waypoint_spacing", first_dataset.waypoint_spacing),
                    metric_waypoint_spacing=viz_dataset_config.get("metric_waypoint_spacing", first_dataset.metric_waypoint_spacing),
                    min_dist_cat=first_dataset.min_dist_cat,
                    max_dist_cat=first_dataset.max_dist_cat,
                    min_action_distance=first_dataset.min_action_distance,
                    max_action_distance=first_dataset.max_action_distance,
                    negative_mining=viz_dataset_config.get("negative_mining", first_dataset.negative_mining),
                    len_traj_pred=first_dataset.len_traj_pred,
                    learn_angle=first_dataset.learn_angle,
                    context_size=first_dataset.context_size,
                    end_slack=viz_dataset_config.get("end_slack", 0),
                    goals_per_obs=viz_dataset_config.get("goals_per_obs", 1),
                    normalize=first_dataset.normalize,
                    build_image_cache=False  # Skip LMDB cache building for visualization dataset
                )

                # Log detailed information about the loaded dataset
                print(f"✓ Train visualization dataset loaded successfully:")
                print(f"  - Dataset folder: {train_viz_folder}")
                print(f"  - Number of trajectories: {len(train_viz_dataset.traj_names)}")
                print(f"  - Number of samples: {len(train_viz_dataset)}")
                print(f"  - Image LMDB cache: {'DISABLED' if not train_viz_dataset.build_image_cache else 'ENABLED'} (loads images directly from filesystem)")
                print(f"  - Trajectory names: {[os.path.basename(traj) for traj in train_viz_dataset.traj_names[:5]]}{'...' if len(train_viz_dataset.traj_names) > 5 else ''}")

            except Exception as e:
                print(f"✗ Warning: Failed to load train visualization dataset: {e}")
                print(f"  - Attempted to load from: {train_viz_folder}")
                train_viz_dataset = None

            # Create train visualization dataloader
            if train_viz_dataset is not None:
                train_viz_dataloader = DataLoader(
                    train_viz_dataset,
                    batch_size=num_images_log,
                    shuffle=True,
                    num_workers=0,  # Use 0 workers to avoid issues
                    pin_memory=True
                )
                print(f"Created train visualization dataloader with {len(train_viz_dataset)} samples")
            else:
                train_viz_dataloader = None
                print("Train visualization dataloader not created due to dataset loading failure")

            # Create test visualization dataset using ViNT_Dataset (not FeatureDataset)
            test_viz_folder = os.path.join(viz_dataset_config["data_folder"], "test_viz")
            print(f"Loading test visualization dataset from: {test_viz_folder}")

            try:
                test_viz_dataset = ViNT_Dataset(
                    data_folder=test_viz_folder,
                    split="train",  # Use "train" split but load from test_viz folder
                    split_ratio=1.0,  # Use all trajectories for visualization
                    dataset_name=f"{viz_dataset_name}_test_viz",  # Unique name for viz dataset
                    dataset_index=0,  # Use 0 for visualization
                    image_size=first_dataset.image_size,
                    waypoint_spacing=viz_dataset_config.get("waypoint_spacing", first_dataset.waypoint_spacing),
                    metric_waypoint_spacing=viz_dataset_config.get("metric_waypoint_spacing", first_dataset.metric_waypoint_spacing),
                    min_dist_cat=first_dataset.min_dist_cat,
                    max_dist_cat=first_dataset.max_dist_cat,
                    min_action_distance=first_dataset.min_action_distance,
                    max_action_distance=first_dataset.max_action_distance,
                    negative_mining=viz_dataset_config.get("negative_mining", first_dataset.negative_mining),
                    len_traj_pred=first_dataset.len_traj_pred,
                    learn_angle=first_dataset.learn_angle,
                    context_size=first_dataset.context_size,
                    end_slack=viz_dataset_config.get("end_slack", 0),
                    goals_per_obs=viz_dataset_config.get("goals_per_obs", 1),
                    normalize=first_dataset.normalize,
                    build_image_cache=False  # Skip LMDB cache building for visualization dataset
                )

                # Log detailed information about the loaded dataset
                print(f"✓ Test visualization dataset loaded successfully:")
                print(f"  - Dataset folder: {test_viz_folder}")
                print(f"  - Number of trajectories: {len(test_viz_dataset.traj_names)}")
                print(f"  - Number of samples: {len(test_viz_dataset)}")
                print(f"  - Image LMDB cache: {'DISABLED' if not test_viz_dataset.build_image_cache else 'ENABLED'} (loads images directly from filesystem)")
                print(f"  - Trajectory names: {[os.path.basename(traj) for traj in test_viz_dataset.traj_names[:5]]}{'...' if len(test_viz_dataset.traj_names) > 5 else ''}")

            except Exception as e:
                print(f"✗ Warning: Failed to load test visualization dataset: {e}")
                print(f"  - Attempted to load from: {test_viz_folder}")
                test_viz_dataset = None

            # Create test visualization dataloader for each dataset type
            if test_viz_dataset is not None:
                for dataset_type in test_dataloaders:
                    test_viz_dataloaders[dataset_type] = DataLoader(
                        test_viz_dataset,
                        batch_size=num_images_log,
                        shuffle=True,
                        num_workers=0,  # Use 0 workers to avoid issues
                        pin_memory=True
                    )
                print(f"Created test visualization dataloader with {len(test_viz_dataset)} samples")
            else:
                print("Test visualization dataloader not created due to dataset loading failure")

        # Summary of visualization dataset loading
        print("\n" + "="*60)
        print("VISUALIZATION DATASET SUMMARY")
        print("="*60)
        if train_viz_dataloader is not None:
            print(f"✓ Train visualization: {len(train_viz_dataset)} samples from {len(train_viz_dataset.traj_names)} trajectories")
        else:
            print("✗ Train visualization: Not loaded")

        if test_viz_dataloaders:
            print(f"✓ Test visualization: {len(test_viz_dataset)} samples from {len(test_viz_dataset.traj_names)} trajectories")
        else:
            print("✗ Test visualization: Not loaded")
        print("="*60 + "\n")

    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            print(f"Start ViNT DP Training Epoch {epoch}/{current_epoch + epochs - 1}")
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
                viz_dataloader=train_viz_dataloader,  # Pass visualization dataloader
            )
            lr_scheduler.step()

        numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
        torch.save(ema_model.averaged_model.state_dict(), numbered_path)
        numbered_path = os.path.join(project_folder, f"ema_latest.pth")
        print(f"Saved EMA model to {numbered_path}")

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(model.state_dict(), numbered_path)
        torch.save(model.state_dict(), latest_path)
        print(f"Saved model to {numbered_path}")

        # save optimizer
        numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
        latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
        torch.save(optimizer.state_dict(), latest_optimizer_path)

        # save scheduler
        numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
        latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
        torch.save(lr_scheduler.state_dict(), latest_scheduler_path)

        if (epoch + 1) % eval_freq == 0:
            for dataset_type in test_dataloaders:
                print(
                    f"Start {dataset_type} ViNT DP Testing Epoch {epoch}/{current_epoch + epochs - 1}"
                )
                loader = test_dataloaders[dataset_type]
                evaluate_nomad(
                    eval_type=dataset_type,
                    ema_model=ema_model,
                    dataloader=loader,
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
                    viz_dataloader=test_viz_dataloaders.get(dataset_type),  # Pass visualization dataloader
                )
        if use_mlflow:
            # Use epoch as step for epoch-level metrics
            mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)

        if lr_scheduler is not None:
            lr_scheduler.step()

    print()


def load_model(model, model_type, checkpoint: dict) -> None:
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
