import os
import json
import torch
import numpy as np
from typing import Tuple

from visualnav_transformer.train.vint_train.data.base_dataset import BaseViNTDataset
from visualnav_transformer.train.vint_train.data.data_utils import (
    find_max_goal_distance_meters
)

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
        image_size: Tuple[int, int],
        waypoint_spacing: int,
        min_goal_distance_meteres: float,
        max_goal_distance_meters: float,
        negative_mining: bool,
        len_traj_pred: int,
        context_size: int,
        end_slack: int = 0,
        goals_per_obs: int = 1,
        normalize: bool = True,
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
            image_size: Size of the images (width, height)
            waypoint_spacing: Spacing between waypoints
            metric_waypoint_spacing: Metric spacing between waypoints
            min_dist_cat: Minimum distance category
            max_dist_cat: Maximum distance category
            min_action_distance: Minimum action distance
            max_action_distance: Maximum action distance
            negative_mining: Whether to use negative mining
            len_traj_pred: Length of trajectory prediction
            learn_angle: Whether to learn angle
            context_size: Number of context frames
            end_slack: End slack
            goals_per_obs: Number of goals per observation
            normalize: Whether to normalize
        """

        # Call parent constructor with all shared parameters
        super().__init__(
            data_folder=data_folder,
            split=split,
            split_ratio=split_ratio,
            dataset_name=dataset_name,
            dataset_index=dataset_index,
            image_size=image_size,
            waypoint_spacing=waypoint_spacing,
            min_goal_distance_meteres=min_goal_distance_meteres,
            max_goal_distance_meters=max_goal_distance_meters,
            negative_mining=negative_mining,
            len_traj_pred=len_traj_pred,
            context_size=context_size,
            end_slack=end_slack,
            goals_per_obs=goals_per_obs,
            normalize=normalize,
        )
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

        # Map trajectory names to their feature chunks
        self.traj_to_chunks = {}
        for chunk_id in self.feature_chunks:
            # Extract trajectory name from chunk ID (e.g., "traj_name_00000" -> "traj_name")
            parts = chunk_id.split("_")
            traj_name = "_".join(parts[:-1])
            if traj_name not in self.traj_to_chunks:
                self.traj_to_chunks[traj_name] = []
            self.traj_to_chunks[traj_name].append(chunk_id)

        # Filter trajectories to only include those with feature chunks
        self.traj_names = [traj for traj in self.traj_names if os.path.basename(traj) in self.traj_to_chunks]

    def _build_index(self):
        """
        Build index with meter-based max_goal_distance calculation.
        Similar to ViNT_Dataset logic but using actual distances instead of frame counts.
        """
        samples_index = []
        goals_index = []
        
        for traj_path in self.traj_names:
            base = os.path.basename(traj_path)
            chunk_ids = self.traj_to_chunks.get(base, [])
            if not chunk_ids:
                continue

            for chunk_id in chunk_ids:
                chunk_pt = os.path.join(self.feature_folder, f"{chunk_id}.pt")
                cd = torch.load(chunk_pt, map_location="cpu")
                
                # Extract data from chunk
                # chunk position format: [{'position': [x,y,z], 'yaw': float}, ...) ]
                traj_positions = [np.array(pos["position"][0], pos["position"][1], pos["yaw"]) 
                                  for pos in cd["positions"]]
                traj_data = dict(positions=traj_positions) # 
                chunk_len = len(traj_positions)

                # Build goals_index
                for local_t in range(chunk_len):
                    goals_index.append((chunk_id, local_t))

                # Compute obs frame bounds (same as original)
                begin_local = self.context_size * self.waypoint_spacing
                end_local = chunk_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing

                # For each valid obs frame
                for local_i in range(begin_local, end_local):                    
                    max_m = self.max_goal_distance_meters
                    max_goal_dist_frames = find_max_goal_distance_meters(traj_data, local_i, max_m)                    
                    samples_index.append((chunk_id, local_i, max_goal_dist_frames))

        return samples_index, goals_index


    def _load_feature(self, chunk_id: str, time: int):
        """Load feature for a specific trajectory and time with comprehensive error handling."""
        path = os.path.join(self.feature_folder, f"{chunk_id}.pt")
        # Add comprehensive error handling for chunk loading
        if not os.path.exists(path):
            raise FileNotFoundError(f"Chunk file does not exist: {path}")
        
        # Try loading with different methods for robustness
        try:
            chunk_data = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e1:
            raise RuntimeError(f"Warning: Failed to load chunk with weights_only=False: {e1}")
        
        # Validate chunk data structure
        if not isinstance(chunk_data, dict):
            raise ValueError(f"Error: Invalid chunk data format for {chunk_id}: expected dict, got {type(chunk_data)}")
        
        if "features" not in chunk_data.keys():
            raise ValueError(f"Error: Missing key 'features' in chunk {chunk_id}")
        
        # Validate feature tensor
        features = chunk_data["features"]
        if not isinstance(features, torch.Tensor):
            raise ValueError(f"Error: Features in chunk {chunk_id} are not a tensor: {type(features)}")
        
        if features.dim() != 2:
            raise ValueError(f"Error: Features in chunk {chunk_id} have wrong dimensions: {features.shape}")
        
        if features.shape[1] != self.feature_dim:
            raise ValueError(f"Warning: Feature dimension mismatch in chunk {chunk_id}: expected {self.feature_dim}, got {features.shape[1]}")
        
        if time >= len(features):
            raise IndexError(f"Error: Index {time} out of bounds for features tensor of length {len(features)} in chunk {chunk_id}")
        
        return features[time].float()  # up-cast to fp32
    
    
    def _load_samples(self, chunk_id: str, time: int, chunk_goal: str, goal_time: int):
        """Load context features for a specific trajectory and time."""
        context_times = self._get_context_times(time)
        context_features = [self._load_feature(chunk_id, t) for t in context_times]
        obs_features = torch.stack(context_features)
        goal_feature = self._load_feature(chunk_goal, goal_time)
        return obs_features, goal_feature
    