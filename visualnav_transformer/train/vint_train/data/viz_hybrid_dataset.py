import os
import torch
import numpy as np
from visualnav_transformer.train.vint_train.data.feature_dataset import FeatureDataset
from visualnav_transformer.train.vint_train.data.data_utils import (
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
        image_size=(240, 320),
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
        goals_per_obs=1
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
        super().__init__(
            data_folder=viz_folder,
            feauture_folder=feature_folder,
            split=split,
            split_ratio=split_ratio,
            dataset_name=dataset_name,
            dataset_index=dataset_index,
            image_size=image_size,
            waypoint_spacing=waypoint_spacing,
            min_goal_distance_meteres=min_goal_distance_meters,
            max_goal_distance_meters=max_goal_distance_meters,
            negative_mining=negative_mining,
            len_traj_pred=len_traj_pred,
            context_size=context_size,
            end_slack=end_slack,
            goals_per_obs=goals_per_obs,
            normalize=normalize,
        )
        

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
                global_indices = cd["indices"]

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
            trajectory_name, chunk_id, goal_time, local_i = self._sample_negative()
            return trajectory_name, chunk_id, goal_time, local_i, True
        else:
            offset_frames = int(goal_offset * self.waypoint_spacing)
            goal_time = curr_time + offset_frames
            local_i = local_i + offset_frames
            return traj_name, chunk_id, goal_time, local_i, False
        
    def _load_samples(self, trajectory_name, time, goal_trajectory_name, goal_time):
        traj_name, chunk_id = trajectory_name
        goal_traj_name, goal_chunk_id = goal_trajectory_name
        time, local_i = time
        goal_time, goal_local_i = goal_time
        
        context_times_features = self._get_context_times(local_i)
        context_times_images = self._get_context_times(time)
        context_features = [self._load_feature(chunk_id, t) for t in context_times_features]
        context_images = [self._load_image(traj_name, t) for t in context_times_images]
        obs_features = torch.stack(context_features)
        obs_images = torch.cat(context_images)
        goal_feature = self._load_feature(goal_chunk_id, goal_local_i)
        goal_image = self._load_image(goal_traj_name, goal_local_i)
        
        return obs_images, goal_image, obs_features, goal_feature
    

    def __getitem__(self, idx):
        """
        Get sample with proper FP16/FP32 handling and error recovery.
        Extended version that returns both images and features.
        """
        f_curr, curr_time, chunk_id, local_i, max_goal_dist = self.index_to_data[idx]
        f_goal, goal_time, goal_chunk_id, goal_local_i, goal_is_negative = self._sample_goal(
            (f_curr, chunk_id), (curr_time, local_i), max_goal_dist
        )

        # Load observation features and goal features
        obs_image, goal_image, obs_features, goal_feature = self._load_samples(
            (f_curr, chunk_id), (curr_time, local_i), (f_goal, goal_chunk_id), (goal_time, goal_local_i)
        )

        # Load trajectory data
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        goal_traj_data = self._get_trajectory(f_goal)
        goal_traj_len = len(goal_traj_data["position"])
        assert goal_time < goal_traj_len, f"{goal_time} an {goal_traj_len}"

        # Compute actions using base class method
        actions, goal_pos, distance, action_mask = self._compute_actions(
            curr_traj_data, curr_time, goal_time, goal_is_negative
        )

        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        actions_torch = calculate_sin_cos(actions_torch)

        return (
            torch.as_tensor(obs_image, dtype=torch.float32),  # [12, H, W] - for visualization display
            torch.as_tensor(goal_image, dtype=torch.float32),  # [3, H, W] - for visualization display
            actions_torch,  # [5, 4] - action labels
            torch.as_tensor(distance, dtype=torch.int64),  # distance label
            torch.as_tensor(goal_pos, dtype=torch.float32),  # goal position
            torch.as_tensor(self.dataset_index, dtype=torch.int64),  # dataset index
            torch.as_tensor(action_mask, dtype=torch.float32),  # action mask
            obs_features,  # [4, 1024] - ADDITIONAL: context features for model (FP32)
            goal_feature,  # [1024] - ADDITIONAL: goal feature for model (FP32)
        )