import os
import torch
from torch.utils.data import DataLoader

# Import dataset classes
from visualnav_transformer.train.vint_train.data.feature_dataset import FeatureDataset
from visualnav_transformer.train.vint_train.data.viz_hybrid_dataset import VizHybridDataset

# Visualization utilities
from visualnav_transformer.train.vint_train.visualizing.visualize_utils import to_numpy
from visualnav_transformer.train.vint_train.visualizing.distance_utils import visualize_dist_pred
from visualnav_transformer.train.vint_train.visualizing.action_utils import visualize_traj_pred

# --- CONFIGURATION ---

# Path to your data_config.yaml (already loaded in your workspace)
scand_config = {
    "data_folder": "/app/Sati_data/SCAND_320x240",
    "feature_folder": "/app/Sati_data/SCAND_320x240/dino_cache_large",  # adjust if needed
    "split": "train",
    "split_ratio": 1.0,
    "dataset_name": "scand",
    "dataset_index": 0,
    "image_size": (240, 320),
    "waypoint_spacing": 1,
    "metric_waypoint_spacing": 0.38,
    "min_dist_cat": 1,
    "max_dist_cat": 10,
    "min_action_distance": 1,
    "max_action_distance": 10,
    "negative_mining": True,
    "len_traj_pred": 5,
    "learn_angle": True,
    "context_size": 4,
    "end_slack": 0,
    "goals_per_obs": 1,
    "normalize": True,
}

# --- DATASET INITIALIZATION ---

# FeatureDataset
feature_dataset = FeatureDataset(
    data_folder=scand_config["data_folder"],
    feature_folder=scand_config["feature_folder"],
    split=scand_config["split"],
    split_ratio=scand_config["split_ratio"],
    dataset_name=scand_config["dataset_name"],
    dataset_index=scand_config["dataset_index"],
    image_size=scand_config["image_size"],
    waypoint_spacing=scand_config["waypoint_spacing"],
    metric_waypoint_spacing=scand_config["metric_waypoint_spacing"],
    min_dist_cat=scand_config["min_dist_cat"],
    max_dist_cat=scand_config["max_dist_cat"],
    min_action_distance=scand_config["min_action_distance"],
    max_action_distance=scand_config["max_action_distance"],
    negative_mining=scand_config["negative_mining"],
    len_traj_pred=scand_config["len_traj_pred"],
    learn_angle=scand_config["learn_angle"],
    context_size=scand_config["context_size"],
    end_slack=scand_config["end_slack"],
    goals_per_obs=scand_config["goals_per_obs"],
    normalize=scand_config["normalize"],
)



# --- DATALOADER ---

# Only check one sample
sample = feature_dataset[0]

# Visualization for a single sample (dummy predictions for demonstration)
obs_image = to_numpy(sample['obs_image']).reshape(1, *sample['obs_image'].shape)
goal_image = to_numpy(sample['goal_image']).reshape(1, *sample['goal_image'].shape)
dist_pred = torch.rand(1) * 10  # Dummy prediction
dist_label = torch.rand(1) * 10  # Dummy label

visualize_dist_pred(
    batch_obs_images=obs_image,
    batch_goal_images=goal_image,
    batch_dist_preds=dist_pred.numpy(),
    batch_dist_labels=dist_label.numpy(),
    eval_type="scand_test_single",
    save_folder="./test_viz",
    epoch=0,
    num_images_preds=1,
    use_mlflow=False,
    display=True,
)

# Example: visualize trajectory predictions (dummy values for demonstration)
goal = torch.rand(1, 2)
pred_waypoints = torch.rand(1, scand_config["len_traj_pred"], 2)
label_waypoints = torch.rand(1, scand_config["len_traj_pred"], 2)
dataset_indices = torch.zeros(1, dtype=torch.int)

visualize_traj_pred(
    batch_obs_images=obs_image,
    batch_goal_images=goal_image,
    dataset_indices=dataset_indices.numpy(),
    batch_goals=goal.numpy(),
    batch_pred_waypoints=pred_waypoints.numpy(),
    batch_label_waypoints=label_waypoints.numpy(),
    eval_type="scand_test_single",
    normalized=True,
    save_folder="./test_viz",
    epoch=0,
    num_images_preds=1,
    use_mlflow=False,
    display=True,
)

print("Single sample visualization complete.")
