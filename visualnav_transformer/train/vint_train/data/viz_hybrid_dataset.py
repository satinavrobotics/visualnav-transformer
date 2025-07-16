import os
import json
import torch
import numpy as np
from visualnav_transformer.train.vint_train.data.dataset.feature_dataset import FeatureDataset
from visualnav_transformer.train.vint_train.data.dataset.data_utils import (
    calculate_sin_cos,
    find_max_goal_distance_meters,
)

class VizHybridDataset(FeatureDataset):
    """
    Hybrid dataset for visualization that returns both images and their corresponding DINO features.
    
    This ensures perfect alignment between displayed images and model predictions.
    
    Structure:
    - Images: From viz_folder/lmdb/ (built by build_viz_cache.py)
    - Features: From dino_cache_folder/lmdb_cache/ (built by dino_cache.py)
    - Trajectory: From traj_data.json files
    """
    
    def __init__(
        self,
        viz_folder,
        feature_folder,
        context_size=4,
        waypoint_spacing=1,
        len_traj_pred=5,
        normalize=True,
        split="train",
        split_ratio=1.0,
        dataset_name="viz_hybrid",
        dataset_index=0,
        min_goal_distance_meters=1.0,
        max_goal_distance_meters=10.3,
        negative_mining=False,
        end_slack=0,
        force_rebuild_indices=False,
    ):
        """
        Args:
            viz_folder: Path to viz folder (e.g., /app/Sati_data/Viz_data/train_viz)
            dino_cache_folder: Path to DINO cache (e.g., /app/Sati_data/Viz_data/train_viz/dino_cache_large)
            image_size: Image size for loading
            context_size: Number of context frames (must match training)
            waypoint_spacing: Spacing between waypoints
            len_traj_pred: Length of trajectory prediction
            learn_angle: Whether to learn angles
            normalize: Whether to normalize actions
            metric_waypoint_spacing: Metric spacing between waypoints
            split: Dataset split ('train' or 'test')
            split_ratio: Ratio of data to use for training
            dataset_name: Name of the dataset
            dataset_index: Index of the dataset
            min_goal_distance_meters: Minimum goal distance in meters
            max_goal_distance_meters: Maximum goal distance in meters
            negative_mining: Whether to use negative mining
            end_slack: End slack for trajectory sampling
            goals_per_obs: Number of goals per observation
        """
        # Call parent constructor
        self.traj_names = None
        self.feature_chunks = None
        super().__init__(
            data_folder=viz_folder,
            feature_folder=feature_folder,
            split=split,
            split_ratio=split_ratio,
            dataset_name=dataset_name,
            dataset_index=dataset_index,
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

        # Map trajectory names to their feature chunks
        self.traj_to_chunks = {}
        for chunk_id in self.feature_chunks:
            # Extract trajectory name from chunk ID (e.g., "traj_name_00000" -> "traj_name")
            parts = chunk_id.split("_")
            traj_name = "_".join(parts[:-1])
            if traj_name not in self.traj_to_chunks:
                self.traj_to_chunks[traj_name] = []
            self.traj_to_chunks[traj_name].append(chunk_id)

        # Get trajectory names from the original dataset
        traj_names_file = os.path.join(viz_folder, "dataset_metadata.json")
        with open(traj_names_file, "r") as f:
            json_data = json.load(f)
        trajectories = json_data["trajectories"]
        if split != "train":
            split_ratio = 1.0 - split_ratio
        split_point = int(len(trajectories) * split_ratio)
        trajectories = trajectories[:split_point] if split == "train" else trajectories[split_point:]
        self.traj_names = [os.path.join(viz_folder, traj["path"]) for traj in trajectories]
        self.traj_names = [traj for traj in self.traj_names if os.path.basename(traj) in self.traj_to_chunks]
        
        self._load_index(force_rebuild=True)


    def _load_chunk_data(self, chunk_id):
        """
        Load chunk data once and cache it to avoid repeated loading.
        Returns (features, positions, yaws, global_indices).
        """
        if chunk_id not in self.chunk_cache:
            chunk_pt = os.path.join(self.feature_folder, f"{chunk_id}.pt")
            cd = torch.load(chunk_pt, map_location="cpu", weights_only=True)

            # Extract all data we need
            features = cd["features"]
            positions = cd["positions"]
            yaws = cd["yaws"]
            global_indices = cd["indices"]

            # Cache the data
            self.chunk_cache[chunk_id] = {
                'features': features,
                'positions': positions,
                'yaws': yaws,
                'global_indices': global_indices
            }

        cached_data = self.chunk_cache[chunk_id]
        return (
            cached_data['features'],
            cached_data['positions'],
            cached_data['yaws'],
            cached_data['global_indices']
        )

    def _build_index(self):
        """
        Build index with meter-based max_goal_distance calculation.
        Similar to ViNT_Dataset logic but using actual distances instead of frame counts.
        """
        samples_index = []
        goals_index = []
        
        if not self.traj_names or not self.feature_chunks:
            return samples_index, goals_index
        
        for traj_path in self.traj_names:
            base = os.path.basename(traj_path)
            chunk_ids = self.traj_to_chunks.get(base, [])
            if not chunk_ids:
                continue

            for chunk_id in chunk_ids:
                chunk_pt = os.path.join(self.feature_folder, f"{chunk_id}.pt")
                cd = torch.load(chunk_pt, map_location="cpu", weights_only=True)

                # Extract trajectory data from chunk
                positions = cd["positions"]
                yaws = cd["yaws"]
                chunk_len = len(positions)
                global_indices = cd["indices"]

                # Build trajectory data dict for distance calculations
                traj_data = dict(position=positions, yaw=yaws)

                # Build goals_index
                for local_t in range(chunk_len):
                    goals_index.append((traj_path, global_indices[local_t], chunk_id, local_t))

                # Compute obs frame bounds (same as original)
                begin_local = self.context_size * self.waypoint_spacing
                end_local = chunk_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing

                # For each valid obs frame
                for local_i in range(begin_local, end_local):
                    max_m = self.max_goal_distance_meters
                    max_goal_dist_frames = find_max_goal_distance_meters(traj_data, local_i, max_m)
                    samples_index.append((
                        traj_path, global_indices[local_i],
                        chunk_id, local_i,
                        max_goal_dist_frames
                    ))

        return samples_index, goals_index
    
    def _sample_goal(self, trajectory_name, curr_time, max_goal_dist):
        """
        Sample a goal from the future in the same trajectory.
        
        Args:
            trajectory_name: Name of the current trajectory
            curr_time: Current time step
            max_goal_dist: Maximum goal distance
            
        Returns:
            Tuple of (trajectory_name, goal_time, goal_is_negative)
        """
        traj_name, chunk_id = trajectory_name
        curr_time, local_i = curr_time
        
        goal_offset = np.random.randint(0, max_goal_dist + 1)
        if goal_offset == 0:
            trajectory_name, goal_time, chunk_id, local_i = self._sample_negative()
            return trajectory_name, chunk_id, goal_time, local_i, True
        else:
            offset_frames = int(goal_offset * self.waypoint_spacing)
            goal_time = curr_time + offset_frames
            local_i = local_i + offset_frames
            return traj_name, chunk_id, goal_time, local_i, False


    def __getitem__(self, idx):
        """
        Get sample with proper FP16/FP32 handling and error recovery.
        Extended version that returns both images and features.
        Optimized to load each chunk only once per call.
        """
        f_curr, curr_time, chunk_id, local_i, max_goal_dist = self.index_to_data[idx]
        f_goal, goal_chunk_id, goal_time, goal_local_i, goal_is_negative = self._sample_goal(
            (f_curr, chunk_id), (curr_time, local_i), max_goal_dist
        )

        # Load current chunk data once and extract everything we need
        curr_features, curr_positions, curr_yaws = self._load_features(chunk_id)
        curr_traj_data = dict(position=curr_positions, yaw=curr_yaws)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        # Load context features from current chunk
        context_times_features = self._get_context_times(local_i)
        obs_features = torch.stack([curr_features[t] for t in context_times_features])

        # Handle goal chunk data
        if goal_chunk_id == chunk_id:
            # Same chunk, reuse loaded data
            goal_traj_data = curr_traj_data
            goal_feature = curr_features[goal_local_i]
        else:
            # Different chunk, load once
            goal_features, goal_positions, goal_yaws = self._load_features(goal_chunk_id)
            goal_traj_data = dict(position=goal_positions, yaw=goal_yaws)
            goal_feature = goal_features[goal_local_i]

        goal_traj_len = len(goal_traj_data["position"])
        assert goal_time < goal_traj_len, f"{goal_time} an {goal_traj_len}"

        # Load images
        context_times_images = self._get_context_times(curr_time)
        context_images = [self._load_image(f_curr, t) for t in context_times_images]
        obs_images = torch.cat(context_images)
        goal_image = self._load_image(f_goal, goal_time)

        # Compute actions using base class method
        actions, goal_pos, distance, action_mask = self._compute_actions(
            curr_traj_data, curr_time, goal_time, goal_is_negative
        )

        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        actions_torch = calculate_sin_cos(actions_torch)

        return (
            torch.as_tensor(obs_images, dtype=torch.float32),  # [12, H, W] - for visualization display
            torch.as_tensor(goal_image, dtype=torch.float32),  # [3, H, W] - for visualization display
            actions_torch,  # [5, 4] - action labels
            torch.as_tensor(distance, dtype=torch.int64),  # distance label
            torch.as_tensor(goal_pos, dtype=torch.float32),  # goal position
            torch.as_tensor(self.dataset_index, dtype=torch.int64),  # dataset index
            torch.as_tensor(action_mask, dtype=torch.float32),  # action mask
            obs_features.float(),  # [4, 1024] - ADDITIONAL: context features for model (FP32)
            goal_feature.float(),  # [1024] - ADDITIONAL: goal feature for model (FP32)
        )