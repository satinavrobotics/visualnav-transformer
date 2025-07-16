import itertools
import os
from typing import Optional, Dict
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
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import gc  # 🧹 Added for RAM/memory cleanup

from visualnav_transformer.train.vint_train.data.dataset.data_utils import (
    VISUALIZATION_IMAGE_SIZE,
)
from visualnav_transformer.train.vint_train.logging.logger import Logger
from visualnav_transformer.train.vint_train.logging.cli_formatter import (
    Colors, Symbols, print_info, print_success, print_warning, print_error,
    print_metrics_table, format_metric_line, format_number, format_time,
    create_custom_tqdm_format, TrainingProgressTracker
)
from visualnav_transformer.train.vint_train.logging.enhanced_logger import (
    create_enhanced_loggers, display_epoch_summary
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

# LOAD DATA CONFIG
from visualnav_transformer import ROOT_TRAIN
with open(
        # CHANGE
    # os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r"
    # os.path.join(ROOT_TRAIN, "vint_train/data/data_config.yaml"), "r"
    os.path.join("/app/visualnav-transformer/config/data/data_config.yaml"), "r"
) as f:
    data_config = yaml.safe_load(f)
    
ACTION_STATS = {
    "min": 0.0,
    "max": 10.0,
}


def setup_loggers(eval_type: str, print_log_freq: int = 100):
    """Create enhanced loggers for training/evaluation with grouped display."""
    return create_enhanced_loggers(eval_type, print_log_freq)


def process_batch_data(data, device, transform, using_prebuilt_dino=False):
    """Process batch data consistently for both training and evaluation."""
    (obs_image, goal_image, actions, distance, goal_pos, dataset_idx, action_mask) = data

    if using_prebuilt_dino:
        # With pre-built DINO features, obs_image and goal_image are already feature vectors
        batch_obs_images = obs_image.to(device)
        batch_goal_images = goal_image.to(device)
        batch_viz_obs_images = None
        batch_viz_goal_images = None
    else:
        # Regular processing for raw images
        obs_images = torch.split(obs_image, 3, dim=1)
        batch_obs_images = torch.cat([transform(obs) for obs in obs_images], dim=1).to(device)
        batch_goal_images = transform(goal_image).to(device)
        batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
        batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])

    action_mask = action_mask.to(device)
    distance = distance.float().to(device)
    goal_pos = goal_pos.to(device)

    return {
        'batch_obs_images': batch_obs_images,
        'batch_goal_images': batch_goal_images,
        'batch_viz_obs_images': batch_viz_obs_images,
        'batch_viz_goal_images': batch_viz_goal_images,
        'actions': actions,
        'distance': distance,
        'goal_pos': goal_pos,
        'dataset_idx': dataset_idx,
        'action_mask': action_mask
    }


def compute_pose_loss(model, obsgoal_cond, goal_pos, goal_mask, device):
    """Compute pose prediction loss if pose head exists."""
    if model.pose_head is not None:
        pose_pred = model("pose_head", obsgoal_cond=obsgoal_cond)
        position_loss = nn.functional.mse_loss(pose_pred[:, :2], goal_pos[:, :2])
        yaw_cos_sim = F.cosine_similarity(pose_pred[:, 2:], goal_pos[:, 2:], dim=1)
        yaw_loss = 1.0 - yaw_cos_sim
        pose_loss = position_loss + yaw_loss
        
        pose_loss = (pose_loss * (1 - goal_mask.float())).mean() / (
            1e-2 + (1 - goal_mask.float()).mean()
        )
    else:
        pose_loss = torch.tensor(0.0, device=device)
    return pose_loss


def compute_distance_loss(model, obsgoal_cond, distance, goal_mask, device):
    """Compute distance prediction loss if distance prediction network exists."""
    if model.dist_pred_net is not None:
        dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
        dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance)
        dist_loss = (dist_loss * (1 - goal_mask.float())).mean() / (
            1e-2 + (1 - goal_mask.float()).mean()
        )
    else:
        dist_loss = torch.tensor(0.0, device=device)
    return dist_loss


def cleanup_memory():
    """Perform memory cleanup."""
    torch.cuda.empty_cache()
    gc.collect()


def handle_visualization(using_prebuilt_dino, viz_dataloader, ema_model, noise_scheduler,
                        batch_obs_images, batch_goal_images, batch_viz_obs_images,
                        batch_viz_goal_images, actions, distance, goal_pos, device,
                        mode, project_folder, epoch, num_images_log, use_mlflow,
                        i, num_batches, loggers, dataset_idx, mlflow_log_freq,
                        print_log_freq, image_log_freq, use_latest):
    """Handle visualization logic for both training and evaluation."""
    if using_prebuilt_dino:
        # New VizHybridDataset format
        (viz_obs_img, viz_goal_img, viz_actions, viz_distance,
            viz_goal_pos, viz_dataset_idx, viz_action_mask,
            viz_obs_features, viz_goal_features) = next(viz_dataloader)

        # Move to device
        viz_obs_img = viz_obs_img.to(device)
        viz_goal_img = viz_goal_img.to(device)
        viz_goal_pos = viz_goal_pos.to(device)
        viz_obs_features = viz_obs_features.to(device)  # [batch, 4, 1024]
        viz_goal_features = viz_goal_features.to(device)  # [batch, 1024]

        # Reshape features to match model input format
        batch_size = viz_obs_features.shape[0]

        # Extract visualization images for display
        obs_images = torch.split(viz_obs_img, 3, dim=1)
        viz_obs_images = obs_images[-1]  # Last context frame

        # Call visualization function with ALIGNED data
        visualize_diffusion_action_distribution(
            ema_model,
            noise_scheduler,
            viz_obs_features,  # Use viz features for model
            viz_goal_features,  # Use viz features for model
            viz_obs_images,  # Use viz images for display
            viz_goal_img,  # Use viz images for display
            viz_actions,  # Use viz actions
            viz_distance,  # Use viz distance
            viz_goal_pos,  # Use viz goal positions
            device,
            mode,
            project_folder,
            epoch,
            min(num_images_log, len(viz_obs_images)),
            30,
            use_mlflow,
            # Additional parameters
            i=i,
            num_batches=num_batches,
            loggers=loggers,
            dataset_index=viz_dataset_idx,
            mlflow_log_freq=mlflow_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
            use_latest=use_latest,
        )
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
            mode,
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
            use_latest=use_latest,
        )


def _compute_losses_nomad(
    ema_model,
    noise_scheduler,
    batch_obs_images,
    batch_goal_images,
    batch_dist_label: torch.Tensor,
    batch_action_label: torch.Tensor,
    device: torch.device,
    action_mask: torch.Tensor,
    batch_goal_pos: Optional[torch.Tensor] = None,
):
    """
    Compute losses for distance, action, and pose prediction.
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
    gc_pose = model_output_dict["gc_pose"]

    gc_dist_loss = F.mse_loss(gc_distance, batch_dist_label.unsqueeze(-1))

    # Compute pose loss if goal positions are provided
    if batch_goal_pos is not None and gc_pose is not None:
        position_loss = F.mse_loss(gc_pose[:, :2], batch_goal_pos[:, :2])
        yaw_cos_sim = F.cosine_similarity(gc_pose[:, 2:], batch_goal_pos[:, 2:], dim=1)
        yaw_loss = 1.0 - yaw_cos_sim.mean()
        gc_pose_loss = position_loss + yaw_loss
    else:
        gc_pose_loss = torch.tensor(0.0, device=device)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert (
        uc_actions.shape == batch_action_label.shape
    ), f"{uc_actions.shape} != {batch_action_label.shape}"
    assert (
        gc_actions.shape == batch_action_label.shape
    ), f"{gc_actions.shape} != {batch_action_label.shape}"

    uc_action_loss = action_reduce(
        F.mse_loss(uc_actions, batch_action_label, reduction="none"), action_mask
    )
    gc_action_loss = action_reduce(
        F.mse_loss(gc_actions, batch_action_label, reduction="none"), action_mask
    )

    uc_action_waypts_cos_similairity = action_reduce(
        F.cosine_similarity(uc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1), action_mask
    )
    uc_multi_action_waypts_cos_sim = action_reduce(
        F.cosine_similarity(
            torch.flatten(uc_actions[:, :, :2], start_dim=1),
            torch.flatten(batch_action_label[:, :, :2], start_dim=1),
            dim=-1,
        ), action_mask
    )

    gc_action_waypts_cos_similairity = action_reduce(
        F.cosine_similarity(gc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1), action_mask
    )
    gc_multi_action_waypts_cos_sim = action_reduce(
        F.cosine_similarity(
            torch.flatten(gc_actions[:, :, :2], start_dim=1),
            torch.flatten(batch_action_label[:, :, :2], start_dim=1),
            dim=-1,
        ), action_mask
    )

    results = {
        "uc_action_loss": uc_action_loss,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_similairity,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim,
        "gc_dist_loss": gc_dist_loss,
        "gc_action_loss": gc_action_loss,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_similairity,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim,
        "gc_pose_loss": gc_pose_loss,
    }

    return results



def action_reduce(unreduced_loss: torch.Tensor, action_mask: torch.Tensor):
    """Reduce loss over non-batch dimensions and apply action mask."""
    # Reduce over non-batch dimensions to get loss per batch element
    while unreduced_loss.dim() > 1:
        unreduced_loss = unreduced_loss.mean(dim=-1)
    assert (
        unreduced_loss.shape == action_mask.shape
    ), f"{unreduced_loss.shape} != {action_mask.shape}"
    return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)


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
    Log data to mlflow and print to console using enhanced logger.
    """
    # Use enhanced logger display (legacy logger support removed)
    loggers.display_metrics(epoch, i, num_batches, use_latest)

    # Get data for MLflow
    if use_mlflow and i % mlflow_log_freq == 0 and mlflow_log_freq != 0:
        data_log = loggers.get_mlflow_data()
        for k, v in data_log.items():
            # Additional safety check for NaN/infinite values before MLflow logging
            if not np.isnan(v) and np.isfinite(v):
                global_step = epoch * num_batches + i
                mlflow.log_metric(k, v, step=global_step if mlflow_increment_step else None)
            else:
                print(f"Warning: Skipping MLflow logging for {k}={v} (NaN/infinite value)")

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
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 4)
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
    gc_pose = model("pose_head", obsgoal_cond=obsgoal_cond)

    return {
        "uc_actions": uc_actions,
        "gc_actions": gc_actions,
        "gc_distance": gc_distance,
        "gc_pose": gc_pose,
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
    gc_poses_list = []

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
        gc_poses_list.append(to_numpy(model_output_dict["gc_pose"]))

    # concatenate
    uc_actions_list = np.concatenate(uc_actions_list, axis=0)
    gc_actions_list = np.concatenate(gc_actions_list, axis=0)
    gc_distances_list = np.concatenate(gc_distances_list, axis=0)
    gc_poses_list = np.concatenate(gc_poses_list, axis=0)

    # split into actions per observation
    uc_actions_list = np.split(uc_actions_list, num_images_log, axis=0)
    gc_actions_list = np.split(gc_actions_list, num_images_log, axis=0)
    gc_distances_list = np.split(gc_distances_list, num_images_log, axis=0)
    gc_poses_list = np.split(gc_poses_list, num_images_log, axis=0)

    gc_distances_avg = [np.mean(dist) for dist in gc_distances_list]
    gc_distances_std = [np.std(dist) for dist in gc_distances_list]
    
    gc_poses_avg = [np.mean(pose, axis=0) for pose in gc_poses_list]
    gc_poses_std = [np.std(pose, axis=0) for pose in gc_poses_list]

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
        # Use the average pose prediction
        pose_pred = from_numpy(np.array([gc_poses_avg[img_idx]])).to(device)

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

        # Note: Logging is handled by the main training loop to avoid duplicate output
        # The visualization function focuses only on creating visualizations

        del action_pred, dist_pred, local_dataset_index
        torch.cuda.empty_cache()