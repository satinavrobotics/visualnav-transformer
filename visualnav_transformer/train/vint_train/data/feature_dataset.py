import os
import json
import pickle
import torch
import numpy as np
import lmdb
import glob
from typing import Dict, List, Optional, Tuple, Union

from torch.utils.data import Dataset

from visualnav_transformer.train.vint_train.data.vint_dataset import ViNT_Dataset
from visualnav_transformer.train.vint_train.data.data_utils import to_local_coords

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

class FeatureDataset(Dataset):
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
        metric_waypoint_spacing: float,
        min_dist_cat: int,
        max_dist_cat: int,
        min_action_distance: int,
        max_action_distance: int,
        negative_mining: bool,
        len_traj_pred: int,
        learn_angle: bool,
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
        self.data_folder = data_folder
        self.feature_folder = feature_folder
        self.split = split
        self.split_ratio = split_ratio
        self.dataset_name = dataset_name
        self.dataset_index = dataset_index
        self.image_size = image_size
        self.waypoint_spacing = waypoint_spacing
        self.metric_waypoint_spacing = metric_waypoint_spacing
        self.distance_categories = list(
            range(min_dist_cat, max_dist_cat + 1, self.waypoint_spacing)
        )
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.negative_mining = negative_mining
        if self.negative_mining:
            self.distance_categories.append(-1)
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle
        self.min_action_distance = min_action_distance
        self.max_action_distance = max_action_distance
        self.context_size = context_size
        self.end_slack = end_slack
        self.goals_per_obs = goals_per_obs
        self.normalize = normalize

        # Load metadata for the feature cache
        metadata_path = os.path.join(feature_folder, "cache_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Feature cache metadata not found at {metadata_path}")

        with open(metadata_path, "r") as f:
            self.cache_metadata = json.load(f)

        # Record the feature dimension in __init__
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

        # Get trajectory names from the original dataset
        traj_names_file = os.path.join(self.data_folder, "dataset_metadata.json")
        with open(traj_names_file, "r") as f:
            json_data = json.load(f)
        trajectories = json_data["trajectories"]
        split_point = int(len(trajectories) * split_ratio)
        trajectories = trajectories[:split_point] if split == "train" else trajectories[split_point:]
        self.traj_names = [os.path.join(self.data_folder, traj["path"]) for traj in trajectories]

        # Filter trajectories to only include those with feature chunks
        self.traj_names = [traj for traj in self.traj_names if os.path.basename(traj) in self.traj_to_chunks]

        # Cache for trajectory data (no size limit)
        self.trajectory_cache = {}

        # Load the index (same as ViNT_Dataset)
        self._load_index()

        # Feature cache
        self.feature_cache = {}

        if self.learn_angle:
            self.num_action_params = 3
        else:
            self.num_action_params = 2

        # Flag to indicate this is using pre-built DINO features
        self.prebuilt_dino = True

    def _load_index(self):
        index_pkl = os.path.join(
            self.data_folder,
            f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_context_n{self.context_size}_slack_{self.end_slack}.pkl",
        )
        try:
            with open(index_pkl, "rb") as f:
                self.index_to_data, self.goals_index = pickle.load(f)
        except FileNotFoundError:
            self.index_to_data, self.goals_index = self._build_index_from_json()
            with open(index_pkl, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _build_index_from_json(self):
        """
        Build index with meter-based max_goal_distance calculation.
        Similar to ViNT_Dataset logic but using actual distances instead of frame counts.
        """
        samples_index = []
        goals_index = []
        
        # Configuration: use max_chunk_distance_m from cache metadata instead of hardcoded value
        max_goal_distance_meters = self.cache_metadata.get("max_chunk_distance_m", 10.0)
        print(f"Using max_goal_distance_meters: {max_goal_distance_meters}m from cache metadata")
        
        for traj_path in self.traj_names:
            base = os.path.basename(traj_path)
            chunk_ids = self.traj_to_chunks.get(base, [])
            if not chunk_ids:
                continue

            for chunk_id in chunk_ids:
                chunk_pt = os.path.join(self.feature_folder, f"{chunk_id}.pt")
                cd = torch.load(chunk_pt, map_location="cpu")
                
                # Extract data from chunk
                global_indices = cd["indices"]
                chunk_positions = cd["positions"]  # [{'position': [x,y,z], 'yaw': float}, ...]
                chunk_len = len(global_indices)

                # Build goals_index
                for t in global_indices:
                    goals_index.append((traj_path, t))

                # Compute obs frame bounds (same as original)
                begin_local = self.context_size * self.waypoint_spacing
                end_local = chunk_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing

                # For each valid obs frame
                for local_i in range(begin_local, end_local):
                    global_t = global_indices[local_i]
                    
                    # NEW: Calculate meter-based max goal distance
                    # This replaces: min(self.max_dist_cat * self.waypoint_spacing, traj_len - curr_time - 1)
                    max_goal_dist_frames = self._calculate_max_goal_distance_meters(
                        chunk_positions,
                        local_i,
                        end_local,
                        max_goal_distance_meters
                    )
                    
                    samples_index.append((traj_path, global_t, max_goal_dist_frames))

        return samples_index, goals_index

    def _calculate_max_goal_distance_meters(
        self,
        chunk_positions: List[Dict],
        local_i: int,
        end_local: int,
        max_distance_meters: float
    ) -> int:
        """
        Calculate the maximum goal distance in frames within the chunk, limited by meters.
        Now fully meter-based without mixing frame-based logic.
        
        Args:
            chunk_positions: Position data from chunk [{'position': [x,y,z], 'yaw': float}, ...]
            local_i: Current local index in chunk
            end_local: End boundary of valid samples in chunk
            max_distance_meters: Maximum allowed distance in meters
            
        Returns:
            Maximum goal distance in frames (like original max_goal_dist)
        """
        curr_pos = np.array(chunk_positions[local_i]["position"][:2])  # x, y only
        max_frames = 0
        
        # Search forward from current position within chunk bounds
        for local_offset in range(1, end_local - local_i):
            candidate_local_idx = local_i + local_offset
            
            # Calculate actual distance using chunk positions
            candidate_pos = np.array(chunk_positions[candidate_local_idx]["position"][:2])
            distance_meters = np.linalg.norm(candidate_pos - curr_pos)
            
            if distance_meters <= max_distance_meters:
                max_frames = local_offset  # This becomes our max_goal_dist equivalent
            else:
                break  # Distance exceeded, stop searching (like original logic)
        
        # Return only the meter-based limit (no mixing with frame-based constraints)
        return max_frames

    def _get_trajectory(self, trajectory_name):
        """
        Get trajectory data for a given trajectory name.
        This loads the trajectory data from the original traj_data.json file,
        following the same pattern as ViNT_Dataset for consistency and efficiency.
        """
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]

        # Load trajectory data from the original JSON file (like ViNT_Dataset)
        try:
            traj_data_path = os.path.join(trajectory_name, "traj_data.json")
            with open(traj_data_path, "r") as f:
                traj_data = json.load(f)

            self.trajectory_cache[trajectory_name] = traj_data
            return traj_data

        except FileNotFoundError:
            print(f"Could not find trajectory data at {traj_data_path}")
            return {"position": [], "yaw": []}
        except Exception as e:
            print(f"Error loading trajectory data from {traj_data_path}: {e}")
            return {"position": [], "yaw": []}

    def _load_feature(self, trajectory_name: str, time: int):
        """Load feature for a specific trajectory and time with comprehensive error handling."""
        base = os.path.basename(trajectory_name)
        if base not in self.traj_to_chunks:
            return None

        # choose the chunk that could contain time
        chunks = sorted(self.traj_to_chunks[base], key=lambda s: int(s.split('_')[-1]))
        # pick the *last* window starting at or before `time`
        chunk_id = None
        for c in chunks:
            if int(c.split('_')[-1]) <= time:
                chunk_id = c
            else:
                break
        if chunk_id is None:
            return None

        if chunk_id not in self.feature_cache:
            path = os.path.join(self.feature_folder, f"{chunk_id}.pt")
            try:
                # Add comprehensive error handling for chunk loading
                if not os.path.exists(path):
                    print(f"Warning: Chunk file does not exist: {path}")
                    return torch.zeros(self.feature_dim, dtype=torch.float32)
                
                # Try loading with different methods for robustness
                try:
                    chunk_data = torch.load(path, map_location="cpu", weights_only=False)
                except Exception as e1:
                    print(f"Warning: Failed to load chunk with weights_only=False: {e1}")
                    try:
                        # Fallback: try with weights_only=True
                        chunk_data = torch.load(path, map_location="cpu", weights_only=True)
                    except Exception as e2:
                        print(f"Error: Failed to load chunk {chunk_id} with both methods: {e1}, {e2}")
                        return torch.zeros(self.feature_dim, dtype=torch.float32)
                
                # Validate chunk data structure
                if not isinstance(chunk_data, dict):
                    print(f"Error: Invalid chunk data format for {chunk_id}: expected dict, got {type(chunk_data)}")
                    return torch.zeros(self.feature_dim, dtype=torch.float32)
                
                required_keys = ["features", "indices"]
                for key in required_keys:
                    if key not in chunk_data:
                        print(f"Error: Missing key '{key}' in chunk {chunk_id}")
                        return torch.zeros(self.feature_dim, dtype=torch.float32)
                
                # Validate feature tensor
                features = chunk_data["features"]
                if not isinstance(features, torch.Tensor):
                    print(f"Error: Features in chunk {chunk_id} are not a tensor: {type(features)}")
                    return torch.zeros(self.feature_dim, dtype=torch.float32)
                
                if features.dim() != 2:
                    print(f"Error: Features in chunk {chunk_id} have wrong dimensions: {features.shape}")
                    return torch.zeros(self.feature_dim, dtype=torch.float32)
                
                if features.shape[1] != self.feature_dim:
                    print(f"Warning: Feature dimension mismatch in chunk {chunk_id}: expected {self.feature_dim}, got {features.shape[1]}")
                    # Try to handle dimension mismatch gracefully
                    if features.shape[1] < self.feature_dim:
                        # Pad with zeros
                        padding = torch.zeros(features.shape[0], self.feature_dim - features.shape[1], dtype=features.dtype)
                        features = torch.cat([features, padding], dim=1)
                        chunk_data["features"] = features
                    else:
                        # Truncate
                        features = features[:, :self.feature_dim]
                        chunk_data["features"] = features
                
                self.feature_cache[chunk_id] = chunk_data
                
            except Exception as e:
                print(f"Critical error loading chunk {chunk_id}: {e}")
                return torch.zeros(self.feature_dim, dtype=torch.float32)

        cd = self.feature_cache[chunk_id]
        
        # Additional validation when accessing cached data
        try:
            if time not in cd["indices"]:
                return None
            local_idx = cd["indices"].index(time)
            
            # Bounds checking for local_idx
            if local_idx >= len(cd["features"]):
                print(f"Error: Index {local_idx} out of bounds for features tensor of length {len(cd['features'])} in chunk {chunk_id}")
                return torch.zeros(self.feature_dim, dtype=torch.float32)
            
            feature = cd["features"][local_idx]
            
            # Ensure feature is the right shape and type
            if feature.dim() != 1 or feature.shape[0] != self.feature_dim:
                print(f"Error: Invalid feature shape {feature.shape} for chunk {chunk_id}, index {local_idx}")
                return torch.zeros(self.feature_dim, dtype=torch.float32)
            
            return feature.float()  # up-cast to fp32
            
        except Exception as e:
            print(f"Error accessing feature from chunk {chunk_id} at time {time}: {e}")
            return torch.zeros(self.feature_dim, dtype=torch.float32)

    def _compute_actions(self, curr_traj_data, curr_time, goal_time):
        """
        Compute actions and goal position using the same logic as ViNT_Dataset.
        This ensures proper waypoint trajectories and coordinate transformations.
        Added error handling for edge cases.
        """
        # Ensure goal_time is within trajectory bounds
        max_goal_time = len(curr_traj_data["position"]) - 1
        if goal_time > max_goal_time:
            print(f"Warning: goal_time {goal_time} exceeds trajectory length {len(curr_traj_data['position'])}, clamping to {max_goal_time}")
            goal_time = max_goal_time

        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        
        # Ensure we don't exceed trajectory bounds
        max_end_index = len(curr_traj_data["position"])
        if end_index > max_end_index:
            end_index = max_end_index
            print(f"Warning: Trajectory too short for full prediction, using end_index {end_index}")

        yaw = curr_traj_data["yaw"][start_index : end_index : self.waypoint_spacing]
        positions = curr_traj_data["position"][start_index : end_index : self.waypoint_spacing]

        # Use the clamped goal_time for goal position
        goal_pos = np.array(curr_traj_data["position"][goal_time])

        yaw = np.array(yaw)  # Using np for avoiding list errors
        positions = np.array(positions)

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            const_len = self.len_traj_pred + 1 - yaw.shape[0]
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            positions = np.concatenate(
                [positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0
            )

        assert yaw.shape == (
            self.len_traj_pred + 1,
        ), f"{yaw.shape} and {(self.len_traj_pred + 1,)} should be equal"
        assert positions.shape == (
            self.len_traj_pred + 1,
            2,
        ), f"{positions.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        # CRITICAL FIX: Use proper coordinate transformation like ViNT_Dataset
        waypoints = to_local_coords(positions, positions[0], yaw[0])
        
        # Use the same reference frame for goal position consistency
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])

        assert waypoints.shape == (
            self.len_traj_pred + 1,
            2,
        ), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        if self.learn_angle:
            yaw = yaw[1:] - yaw[0]
            actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
        else:
            actions = waypoints[1:]

        if self.normalize:
            actions[:, :2] /= (self.metric_waypoint_spacing * self.waypoint_spacing)
            goal_pos /= (self.metric_waypoint_spacing * self.waypoint_spacing)

        assert actions.shape == (
            self.len_traj_pred,
            self.num_action_params,
        ), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"

        return actions, goal_pos

    def _sample_goal(self, f_curr, curr_time, max_goal_dist):
        """
        Sample a goal using the same logic as ViNT_Dataset.
        This ensures consistent training behavior between feature and image datasets.
        """
        # Use ViNT_Dataset's goal sampling pattern
        goal_offset = np.random.randint(0, max_goal_dist + 1)
        if goal_offset == 0:
            # Sample a negative goal from different trajectory
            trajectory_name, goal_time = self._sample_negative()
            return trajectory_name, goal_time, True
        else:
            # Sample a positive goal from same trajectory
            goal_time = curr_time + int(goal_offset * self.waypoint_spacing)
            return f_curr, goal_time, False

    def _sample_negative(self):
        """
        Sample a goal from a (likely) different trajectory.
        This matches ViNT_Dataset's negative sampling logic.
        """
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.index_to_data)

    def __getitem__(self, i):
        """
        Get an item from the dataset.

        Returns:
            obs_features: Features for the observation frames
            goal_features: Features for the goal frame
            actions: Action labels
            distance: Distance label
            goal_pos: Goal position
            dataset_index: Dataset index
            action_mask: Action mask
        """
        f_curr, curr_time, max_goal_dist = self.index_to_data[i]
        f_goal, goal_time, goal_is_negative = self._sample_goal(
            f_curr, curr_time, max_goal_dist
        )

        # Load features for context frames
        context_times = list(
            range(
                curr_time + -self.context_size * self.waypoint_spacing,
                curr_time + 1,
                self.waypoint_spacing,
            )
        )

        # Load features for each context frame
        context_features = []
        for t in context_times:
            feature = self._load_feature(f_curr, t)
            if feature is None:
                feature = torch.zeros(self.feature_dim, dtype=torch.float16)
            feature = feature.float()          # <── add
            context_features.append(feature)

        # Stack context features
        obs_features = torch.stack(context_features)

        # Load goal feature
        goal_feature = self._load_feature(f_goal, goal_time)
        if goal_feature is None:
            goal_feature = torch.zeros(self.feature_dim, dtype=torch.float16)
        goal_feature = goal_feature.float()    # <── add

        # Load trajectory data
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        goal_traj_data = self._get_trajectory(f_goal)
        goal_traj_len = len(goal_traj_data["position"])
        assert goal_time < goal_traj_len, f"{goal_time} an {goal_traj_len}"

        # Compute actions
        actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)

        # CRITICAL FIX: Compute distance in waypoint units like ViNT_Dataset
        if goal_is_negative:
            distance = self.max_dist_cat  # Use max distance for negatives like ViNT_Dataset
        else:
            distance = (goal_time - curr_time) // self.waypoint_spacing
            assert (
                goal_time - curr_time
            ) % self.waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}"

        # CRITICAL FIX: Apply proper action masking like ViNT_Dataset
        action_mask = (
            (distance < self.max_action_distance)
            and (distance > self.min_action_distance)
            and (not goal_is_negative)
        )

        # Convert to tensors
        actions = torch.from_numpy(actions).float()
        distance = torch.tensor(distance).long()  # Use long for distance like ViNT_Dataset
        goal_pos = torch.from_numpy(goal_pos).float()
        dataset_index = torch.tensor(self.dataset_index).long()
        action_mask = torch.tensor(action_mask).float()

        return obs_features, goal_feature, actions, distance, goal_pos, dataset_index, action_mask