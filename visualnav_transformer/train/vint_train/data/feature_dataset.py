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
        Build an index consisting of tuples
          (trajectory name, global_time, max_goal_distance)
        where max_goal_distance is measured within each chunk.
        """
        samples_index = []
        goals_index = []

        # for each trajectory (by path), look up its feature chunks
        for traj_path in self.traj_names:
            base = os.path.basename(traj_path)
            chunk_ids = self.traj_to_chunks.get(base, [])
            if not chunk_ids:
                continue

            for chunk_id in chunk_ids:
                # load the .pt to get its list of global frame indices
                chunk_pt = os.path.join(self.feature_folder, f"{chunk_id}.pt")
                cd = torch.load(chunk_pt, map_location="cpu")
                global_indices = cd["indices"]            # e.g. [345,346,…,544]
                chunk_len = len(global_indices)

                # build goals_index for every frame in the chunk
                for t in global_indices:
                    goals_index.append((traj_path, t))

                # compute local bounds to sample obs frames
                begin_local = self.context_size * self.waypoint_spacing
                end_local = chunk_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing

                # for each valid local obs-time, compute its global time and local max distance
                for local_i in range(begin_local, end_local):
                    global_t = global_indices[local_i]
                    # how many steps remain *in this chunk*
                    max_local = end_local - local_i
                    # but also cap to your config’s max_dist_cat
                    max_goal_dist = min(self.max_dist_cat * self.waypoint_spacing, max_local)
                    samples_index.append((traj_path, global_t, max_goal_dist))

        return samples_index, goals_index

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
                self.feature_cache[chunk_id] = torch.load(path, map_location="cpu")
            except Exception as e:
                print(f"load fail {chunk_id}: {e}")
                return None

        cd = self.feature_cache[chunk_id]
        if time not in cd["indices"]:
            return None
        local_idx = cd["indices"].index(time)

        return cd["features"][local_idx].float()     # up-cast to fp32

    def _compute_actions(self, curr_traj_data, curr_time, goal_time):
        """
        Compute actions and goal position.
        This is the same as in ViNT_Dataset.
        """
        # Get the current position
        curr_pos = np.array(curr_traj_data["position"][curr_time])

        # Get the goal position
        goal_pos = np.array(curr_traj_data["position"][goal_time])

        # Compute the actions
        actions = []
        for i in range(self.len_traj_pred):
            # Get the waypoint time
            waypoint_time = min(
                curr_time + (i + 1) * self.waypoint_spacing, len(curr_traj_data["position"]) - 1
            )

            # Get the waypoint position
            waypoint_pos = np.array(curr_traj_data["position"][waypoint_time])

            # Compute the action (waypoint position relative to current position)
            action = waypoint_pos - curr_pos

            # Add the yaw if needed
            if self.learn_angle:
                waypoint_yaw = curr_traj_data["yaw"][waypoint_time]
                action = np.append(action, waypoint_yaw)

            actions.append(action)

        # Convert to numpy array
        actions = np.array(actions)

        return actions, goal_pos

    def _sample_goal(self, f_curr, curr_time, max_goal_dist):
        """
        Sample a goal from the goals index.
        This is the same as in ViNT_Dataset.
        """
        if self.negative_mining and np.random.random() < 0.5:
            # Sample a negative goal
            f_goal = f_curr
            goal_time = curr_time
            goal_is_negative = True
        else:
            # Sample a positive goal
            goal_dist = np.random.choice(self.distance_categories)
            if goal_dist == -1:
                # Sample a negative goal
                f_goal = f_curr
                goal_time = curr_time
                goal_is_negative = True
            else:
                # Sample a positive goal
                if goal_dist > max_goal_dist:
                    goal_dist = max_goal_dist

                # Get the goal time
                goal_time = curr_time + goal_dist

                # Get the goal trajectory
                f_goal = f_curr
                goal_is_negative = False

        return f_goal, goal_time, goal_is_negative

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

        # Compute distance
        if goal_is_negative:
            distance = -1
        else:
            distance = goal_time - curr_time

        # Create action mask
        action_mask = 1.0
        if goal_is_negative:
            action_mask = 0.0

        # Convert to tensors
        actions = torch.from_numpy(actions).float()
        distance = torch.tensor(distance).float()
        goal_pos = torch.from_numpy(goal_pos).float()
        dataset_index = torch.tensor(self.dataset_index).long()
        action_mask = torch.tensor(action_mask).float()

        return obs_features, goal_feature, actions, distance, goal_pos, dataset_index, action_mask