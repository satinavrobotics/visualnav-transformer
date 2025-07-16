import os
import json
import torch
import numpy as np
from typing import Tuple

from .base_dataset import BaseViNTDataset
from .data_utils import find_max_goal_distance_meters, calculate_sin_cos

class FeatureDataset(BaseViNTDataset):
    """
    Dataset class for loading pre-built DINO features.
    This class mimics the interface of ViNT_Dataset but loads pre-computed features instead of images.
    """
    def __init__(
        self,
        data_folder: str,
        feature_folder: str,
        split: str,
        split_ratio: float,
        dataset_name: str,
        dataset_index: int,
        waypoint_spacing: int,
        min_goal_distance_meters: float,
        max_goal_distance_meters: float,
        negative_mining: bool,
        len_traj_pred: int,
        context_size: int,
        end_slack: int = 0,
        normalize: bool = True,
        force_rebuild_indices: bool = False,
    ):
        """
        Initialize the FeatureDataset.

        Args:
            data_folder: Path to the original data folder (for trajectory data)
            feature_folder: Path to the folder containing pre-built DINO features
            split: 'train' or 'test'
            split_ratio: Ratio of data to use for training
            dataset_name: Name of the dataset
            dataset_index: Index of the dataset
            waypoint_spacing: Spacing between waypoints
            min_goal_distance_meters: Minimum goal distance in meters
            max_goal_distance_meters: Maximum goal distance in meters
            negative_mining: Whether to use negative mining
            len_traj_pred: Length of trajectory prediction
            context_size: Number of context frames
            end_slack: End slack
            normalize: Whether to normalize
            force_rebuild_indices: Whether to force rebuild dataset indices
        """
        # Store feature-specific parameters before calling parent
        self.feature_folder = feature_folder

        # Load metadata for the feature cache
        metadata_path = os.path.join(feature_folder, "cache_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Feature cache metadata not found at {metadata_path}")
        with open(metadata_path, "r") as f:
            self.cache_metadata = json.load(f)
    
        # Record the feature dimension
        self.feature_dim = self.cache_metadata["feature_dim"]
        # Get all available feature chunks
        self.feature_chunks = self.cache_metadata.get("chunks", [])
        if not self.feature_chunks:
            raise ValueError("No feature chunks found in cache metadata")
        
        # Call parent constructor with all shared parameters
        super().__init__(
            data_folder=data_folder,
            split=split,
            split_ratio=split_ratio,
            dataset_name=dataset_name,
            dataset_index=dataset_index,
            image_size=(1, 1),
            waypoint_spacing=waypoint_spacing,
            min_goal_distance_meters=min_goal_distance_meters,
            max_goal_distance_meters=max_goal_distance_meters,
            negative_mining=negative_mining,
            len_traj_pred=len_traj_pred,
            context_size=context_size,
            end_slack=end_slack,
            normalize=normalize,
            force_rebuild_indices=force_rebuild_indices,
        )

    def _build_index(self):
        """
        Build index with meter-based max_goal_distance calculation.
        Similar to ViNT_Dataset logic but using actual distances instead of frame counts.
        """
        samples_index = []
        goals_index = []
        
        for chunk_id in self.feature_chunks:
            chunk_pt = os.path.join(self.feature_folder, f"{chunk_id}.pt")
            chunk_data = torch.load(chunk_pt, map_location="cpu", weights_only=True)
            chunk_len = len(chunk_data["positions"])
            goals_index = [(chunk_id, local_t) for local_t in range(chunk_len)]
            
            begin_local = self.context_size * self.waypoint_spacing
            end_local = chunk_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing
            for local_i in range(begin_local, end_local):                    
                max_goal_dist_frames = find_max_goal_distance_meters(
                    dict(position=chunk_data["positions"], yaw=chunk_data["yaws"]), 
                    local_i, self.max_goal_distance_meters)
                if max_goal_dist_frames <= 1:
                    break                
                samples_index.append((chunk_id, local_i, max_goal_dist_frames))

        return samples_index, goals_index



    def _load_features(self, chunk_id: str):
        """Load feature for a specific trajectory and time with comprehensive error handling."""
        path = os.path.join(self.feature_folder, f"{chunk_id}.pt")
        # Add comprehensive error handling for chunk loading
        if not os.path.exists(path):
            raise FileNotFoundError(f"Chunk file does not exist: {path}")
        
        # Try loading with different methods for robustness
        try:
            chunk_data = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e1:
            raise RuntimeError(f"Warning: Failed to load chunk with weights_only=False: {e1}")
        
        # Validate chunk data structure
        if not isinstance(chunk_data, dict):
            raise ValueError(f"Error: Invalid chunk data format for {chunk_id}: expected dict, got {type(chunk_data)}")
        
        if "features" not in chunk_data.keys():
            raise ValueError(f"Error: Missing key 'features' in chunk {chunk_id}")
        if "positions" not in chunk_data.keys():
            raise ValueError(f"Error: Missing key 'positions' in chunk {chunk_id}")
        if "yaws" not in chunk_data.keys():
            raise ValueError(f"Error: Missing key 'yaw' in chunk {chunk_id}")
        
        # Validate feature tensor
        features = chunk_data["features"]
        if not isinstance(features, torch.Tensor):
            raise ValueError(f"Error: Features in chunk {chunk_id} are not a tensor: {type(features)}")
        
        if features.dim() != 2:
            raise ValueError(f"Error: Features in chunk {chunk_id} have wrong dimensions: {features.shape}")
        
        if features.shape[1] != self.feature_dim:
            raise ValueError(f"Warning: Feature dimension mismatch in chunk {chunk_id}: expected {self.feature_dim}, got {features.shape[1]}")
        
        positions = chunk_data["positions"]
        yaws = chunk_data["yaws"]
        
        return features, positions, yaws
    
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to ith datapoint
        Returns:
            Tuple of tensors containing the context, observation, goal, transformed context, transformed observation, transformed goal, distance label, and action label
                obs_image (torch.Tensor): tensor of shape [3, H, W] containing the image of the robot's observation
                goal_image (torch.Tensor): tensor of shape [3, H, W] containing the subgoal image
                dist_label (torch.Tensor): tensor of shape (1,) containing the distance labels from the observation to the goal
                action_label (torch.Tensor): tensor of shape (5, 2) or (5, 4) (if training with angle) containing the action labels from the observation to the goal
                which_dataset (torch.Tensor): index of the datapoint in the dataset [for identifying the dataset for visualization when using multiple datasets]
        """
        curr_chunk_id, curr_time, max_goal_dist = self.index_to_data[i]
        goal_chunk_id, goal_time, goal_is_negative = self._sample_goal(
            curr_chunk_id, curr_time, max_goal_dist
        )
        
        context_times = self._get_context_times(curr_time)
        features, positions, yaws = self._load_features(curr_chunk_id)
        curr_traj_data = dict(position=positions, yaw=yaws,)
        
        obs_features = torch.stack([features[t] for t in context_times])
        if not goal_is_negative:
            goal_feature = features[goal_time]
        else:
            goal_features, _, _ = self._load_features(goal_chunk_id)
            goal_feature = goal_features[goal_time]
            

        # Compute actions using base class method
        actions, goal_pos, distance, action_mask = self._compute_actions(
            curr_traj_data, curr_time, goal_time, goal_is_negative
        )

        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        actions_torch = calculate_sin_cos(actions_torch)

        return (
            torch.as_tensor(obs_features, dtype=torch.float32),
            torch.as_tensor(goal_feature, dtype=torch.float32),
            actions_torch,
            torch.as_tensor(distance, dtype=torch.int64),
            torch.as_tensor(goal_pos, dtype=torch.float32),
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
            torch.as_tensor(action_mask, dtype=torch.float32),
        )
    
    
    
    
    