import os
import json
import pickle
import torch
import numpy as np
import lmdb
import io
from torch.utils.data import Dataset
from visualnav_transformer.train.vint_train.data.data_utils import (
    img_path_to_data,
    get_data_path,
    calculate_sin_cos,
    to_local_coords
)

def calculate_distance_meters(pos1, pos2):
    """Calculate Euclidean distance between two positions in meters."""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def find_max_goal_distance_meters(traj_data, curr_time, max_distance_meters):
    """
    Find the maximum goal distance in frames that corresponds to max_distance_meters.

    Args:
        traj_data: Trajectory data with position information
        curr_time: Current time index
        max_distance_meters: Maximum allowed distance in meters

    Returns:
        Maximum goal distance in frames
    """
    curr_pos = traj_data["position"][curr_time]
    max_goal_frames = 0

    # Search forward from current position
    for future_time in range(curr_time + 1, len(traj_data["position"])):
        future_pos = traj_data["position"][future_time]
        distance_m = calculate_distance_meters(curr_pos, future_pos)

        if distance_m <= max_distance_meters:
            max_goal_frames = future_time - curr_time
        else:
            break  # Stop when we exceed the distance limit

    return max_goal_frames

class VizHybridDataset(Dataset):
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
        dino_cache_folder, 
        image_size=(240, 320),
        context_size=4,
        waypoint_spacing=1,
        len_traj_pred=5,
        learn_angle=True,
        normalize=True,
        metric_waypoint_spacing=0.25
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
        """
        self.viz_folder = viz_folder
        self.dino_cache_folder = dino_cache_folder
        self.image_size = image_size
        self.context_size = context_size
        self.waypoint_spacing = waypoint_spacing
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle
        self.normalize = normalize
        self.metric_waypoint_spacing = metric_waypoint_spacing
        
        # Add missing attributes needed for goal sampling (matching ViNT_Dataset)
        self.max_dist_cat = 10.3  # Default value, should match training config
        self.min_dist_cat = 1   # Default value, should match training config
        
        # Load dataset metadata
        self.metadata_path = os.path.join(viz_folder, "dataset_metadata.json")
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)
        
        # Get trajectory names
        self.traj_names = [os.path.join(viz_folder, traj["path"]) for traj in self.metadata["trajectories"]]
        
        # Open image LMDB cache
        image_lmdb_path = os.path.join(viz_folder, "lmdb")
        if os.path.exists(image_lmdb_path):
            self._image_cache = lmdb.open(image_lmdb_path, readonly=True)
        else:
            raise FileNotFoundError(f"Image LMDB cache not found at {image_lmdb_path}")
        
        # Open feature LMDB cache
        feature_lmdb_path = os.path.join(dino_cache_folder, "lmdb_cache")
        if os.path.exists(feature_lmdb_path):
            self._feature_cache = lmdb.open(feature_lmdb_path, readonly=True)
        else:
            raise FileNotFoundError(f"Feature LMDB cache not found at {feature_lmdb_path}")
        
        # Load feature cache metadata
        cache_metadata_path = os.path.join(dino_cache_folder, "cache_metadata.json")
        with open(cache_metadata_path, "r") as f:
            self.cache_metadata = json.load(f)
        
        # Get feature dimension from cache metadata
        self.feature_dim = self.cache_metadata.get("feature_dim", 1024)  # Default to 1024 for DINO
        
        # Build trajectory to chunks mapping
        self.traj_to_chunks = {}
        for chunk_id in self.cache_metadata["chunks"]:
            # Extract trajectory name from chunk_id
            traj_name = "_".join(chunk_id.split("_")[:-1])
            if traj_name not in self.traj_to_chunks:
                self.traj_to_chunks[traj_name] = []
            self.traj_to_chunks[traj_name].append(chunk_id)
        
        # Build valid sample indices (like ViNT_Dataset)
        self.index_to_data = []
        self.trajectory_cache = {}
        
        for traj_name in self.traj_names:
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])
            begin = self.context_size * self.waypoint_spacing
            end   = traj_len - self.len_traj_pred * self.waypoint_spacing
            for t in range(begin, end):
                # compute meter‐based max_goal_dist for visualization
                mg = find_max_goal_distance_meters(
                traj_data, t,
                self.metric_waypoint_spacing * self.waypoint_spacing
                )
                self.index_to_data.append((traj_name, t, mg))
    
    def _get_trajectory(self, trajectory_name):
        """Load trajectory data from traj_data.json"""
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]
        
        traj_data_path = os.path.join(trajectory_name, "traj_data.json")
        with open(traj_data_path, "r") as f:
            traj_data = json.load(f)
        
        self.trajectory_cache[trajectory_name] = traj_data
        return traj_data
    
    def _load_image(self, trajectory_name, time_idx):
        """Load image from LMDB cache"""
        image_path = get_data_path(self.viz_folder, trajectory_name, time_idx)
        
        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(image_path.encode())
                if image_buffer is None:
                    raise TypeError("Key not found in LMDB cache")
                image_bytes = bytes(image_buffer)
            
            image_bytes = io.BytesIO(image_bytes)
            return img_path_to_data(image_bytes, self.image_size)
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            return torch.zeros(3, self.image_size[0], self.image_size[1])
    
    def _load_feature_from_chunk(self, chunk_id, time_idx, chunk_start_idx):
        """Load feature from a specific chunk with proper FP16/FP32 handling."""
        try:
            with self._feature_cache.begin() as txn:
                chunk_buffer = txn.get(chunk_id.encode())
                if chunk_buffer is None:
                    return torch.zeros(self.feature_dim, dtype=torch.float32)
                
                chunk_data = pickle.loads(chunk_buffer)
                
                # Handle current chunk format
                if "features" in chunk_data:
                    chunk_features = chunk_data["features"]
                    chunk_idx = time_idx - chunk_start_idx
                    
                    if 0 <= chunk_idx < len(chunk_features):
                        feature = chunk_features[chunk_idx]
                        
                        # Ensure consistent FP32 conversion
                        if feature.dtype == torch.float16:
                            feature = feature.float()  # Convert FP16 to FP32
                        elif feature.dtype != torch.float32:
                            feature = feature.to(torch.float32)  # Convert any other dtype to FP32
                        
                        # Ensure feature is 1D and has correct size
                        if feature.dim() > 1:
                            feature = feature.flatten()
                        
                        # Pad or truncate to expected feature dimension
                        if len(feature) != self.feature_dim:
                            if len(feature) < self.feature_dim:
                                # Pad with zeros
                                padding = torch.zeros(self.feature_dim - len(feature), dtype=torch.float32)
                                feature = torch.cat([feature, padding])
                            else:
                                # Truncate to expected size
                                feature = feature[:self.feature_dim]
                        
                        return feature
                
                # Fallback for missing features
                return torch.zeros(self.feature_dim, dtype=torch.float32)
                
        except Exception as e:
            print(f"Failed to load feature chunk {chunk_id} at index {time_idx - chunk_start_idx}: {e}")
            return torch.zeros(self.feature_dim, dtype=torch.float32)

    def _get_feature(self, trajectory_name, time_idx):
        """Get DINO feature for a specific time index with proper error handling."""
        base_traj_name = os.path.basename(trajectory_name)
        
        if base_traj_name not in self.traj_to_chunks:
            print(f"Warning: No chunks found for trajectory {base_traj_name}")
            return torch.zeros(self.feature_dim, dtype=torch.float32)
        
        # Find the chunk that contains this time index
        chunks = sorted(self.traj_to_chunks[base_traj_name],
                       key=lambda x: int(x.split("_")[-1]))
        
        chunk_id = None
        chunk_start = 0
        
        for c in chunks:
            c_start = int(c.split("_")[-1])
            if c_start <= time_idx:
                chunk_id = c
                chunk_start = c_start
            else:
                break
        
        if chunk_id is None:
            print(f"Warning: No chunk found for time index {time_idx} in trajectory {base_traj_name}")
            return torch.zeros(self.feature_dim, dtype=torch.float32)
        
        return self._load_feature_from_chunk(chunk_id, time_idx, chunk_start)

    def _compute_actions(self, traj_data, curr_time, goal_time):
        """Compute actions like ViNT_Dataset"""
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        yaw = traj_data["yaw"][start_index : end_index : self.waypoint_spacing]
        positions = traj_data["position"][start_index : end_index : self.waypoint_spacing]

        goal_pos = np.array(traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)])

        yaw = np.array(yaw)
        positions = np.array(positions)

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            const_len = self.len_traj_pred + 1 - yaw.shape[0]
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            positions = np.concatenate(
                [positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0
            )

        waypoints = to_local_coords(positions, positions[0], yaw[0])
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])

        if self.learn_angle:
            yaw = yaw[1:] - yaw[0]
            actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
        else:
            actions = waypoints[1:]

        if self.normalize:
            actions[:, :2] /= (self.metric_waypoint_spacing * self.waypoint_spacing)
            goal_pos /= (self.metric_waypoint_spacing * self.waypoint_spacing)

        return actions, goal_pos

    def __len__(self):
        return len(self.index_to_data)

    def __getitem__(self, idx):
        """
        Get sample with proper FP16/FP32 handling and error recovery.
        """
        try:
            trajectory_name, curr_time = self.index_to_data[idx]
            traj_data = self._get_trajectory(trajectory_name)

            # Use the same goal sampling logic as FeatureDataset
            # Find maximum goal distance based on meter-based constraints
            # Get max_chunk_distance_m from cache metadata if available, otherwise use default calculation
            if hasattr(self, 'cache_metadata') and 'max_chunk_distance_m' in self.cache_metadata:
                max_distance_meters = self.cache_metadata['max_chunk_distance_m']
            else:
                # Fallback: calculate based on distance categories (meter-based)
                max_distance_meters = self.max_dist_cat * self.metric_waypoint_spacing
            
            max_goal_distance_frames = find_max_goal_distance_meters(traj_data, curr_time, max_distance_meters)
            
            # Also respect trajectory bounds
            traj_len = len(traj_data["position"])
            max_frames_available = traj_len - curr_time - 1
            max_goal_distance_frames = min(max_goal_distance_frames, max_frames_available)
            
            if max_goal_distance_frames <= 0:
                # If no valid goal found within distance constraint, use minimum offset
                max_goal_distance_frames = min(10, max_frames_available)
            
            # Sample goal with same logic as training: random offset within valid range
            if max_goal_distance_frames > 0:
                goal_offset = np.random.randint(1, max_goal_distance_frames + 1)
                goal_time = curr_time + goal_offset
            else:
                # Fallback: use current time as goal (distance = 0)
                goal_time = curr_time

            # Load context images and features with error handling
            context_times = list(range(
                curr_time - self.context_size * self.waypoint_spacing,
                curr_time + 1,
                self.waypoint_spacing,
            ))

            # Load context images
            context_images = []
            for t in context_times:
                try:
                    img = self._load_image(trajectory_name, max(0, t))
                    if img is None:
                        img = torch.zeros(3, self.image_size[0], self.image_size[1], dtype=torch.float32)
                    context_images.append(img)
                except Exception as e:
                    print(f"Error loading image at time {t}: {e}")
                    context_images.append(torch.zeros(3, self.image_size[0], self.image_size[1], dtype=torch.float32))
            
            obs_image = torch.cat(context_images)  # [12, H, W] for context_size=4

            # Load context features with proper FP16/FP32 handling
            context_features = []
            for t in context_times:
                try:
                    feat = self._get_feature(trajectory_name, max(0, t))
                    # Ensure all features are FP32
                    if feat.dtype != torch.float32:
                        feat = feat.to(torch.float32)
                    context_features.append(feat)
                except Exception as e:
                    print(f"Error loading feature at time {t}: {e}")
                    context_features.append(torch.zeros(self.feature_dim, dtype=torch.float32))
            
            obs_features = torch.stack(context_features)  # [4, 1024]

            # Load goal image and feature with error handling
            try:
                goal_image = self._load_image(trajectory_name, goal_time)
                if goal_image is None:
                    goal_image = torch.zeros(3, self.image_size[0], self.image_size[1], dtype=torch.float32)
            except Exception as e:
                print(f"Error loading goal image at time {goal_time}: {e}")
                goal_image = torch.zeros(3, self.image_size[0], self.image_size[1], dtype=torch.float32)
            
            try:
                goal_feature = self._get_feature(trajectory_name, goal_time)
                # Ensure goal feature is FP32
                if goal_feature.dtype != torch.float32:
                    goal_feature = goal_feature.to(torch.float32)
            except Exception as e:
                print(f"Error loading goal feature at time {goal_time}: {e}")
                goal_feature = torch.zeros(self.feature_dim, dtype=torch.float32)

            # Compute actions and goal position
            actions, goal_pos = self._compute_actions(traj_data, curr_time, goal_time)

            # Compute distance (same as training)
            distance = goal_time - curr_time

            # Convert to tensors (same format as ViNT_Dataset)
            actions_torch = torch.as_tensor(actions, dtype=torch.float32)
            if self.learn_angle:
                actions_torch = calculate_sin_cos(actions_torch)

            return (
                obs_image,  # [12, H, W] - for visualization display
                goal_image,  # [3, H, W] - for visualization display
                actions_torch,  # [5, 4] - action labels
                torch.as_tensor(distance, dtype=torch.int64),  # distance label
                torch.as_tensor(goal_pos, dtype=torch.float32),  # goal position
                torch.as_tensor(0, dtype=torch.int64),  # dataset index
                torch.as_tensor(1.0, dtype=torch.float32),  # action mask
                obs_features,  # [4, 1024] - ADDITIONAL: context features for model (FP32)
                goal_feature,  # [1024] - ADDITIONAL: goal feature for model (FP32)
            )

        except Exception as e:
            print(f"Error in VizHybridDataset.__getitem__ for index {idx}: {e}")
            # Return fallback tensors with proper shapes and dtypes
            return (
                torch.zeros(12, self.image_size[0], self.image_size[1], dtype=torch.float32),  # obs_image
                torch.zeros(3, self.image_size[0], self.image_size[1], dtype=torch.float32),   # goal_image
                torch.zeros(self.len_traj_pred, 4 if self.learn_angle else 2, dtype=torch.float32),  # actions
                torch.tensor(0, dtype=torch.int64),  # distance
                torch.zeros(2, dtype=torch.float32),  # goal_pos
                torch.tensor(0, dtype=torch.int64),   # dataset_index
                torch.tensor(1.0, dtype=torch.float32),  # action_mask
                torch.zeros(self.context_size, self.feature_dim, dtype=torch.float32),  # obs_features
                torch.zeros(self.feature_dim, dtype=torch.float32),  # goal_feature
            )