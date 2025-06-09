import io
import os
import pickle
from typing import Tuple

import lmdb
import numpy as np
import torch
import tqdm
import yaml
from torch.utils.data import Dataset
import json

from visualnav_transformer.train.vint_train.data.data_utils import (
    calculate_sin_cos,
    get_data_path,
    img_path_to_data,
    to_local_coords,
)


class ViNT_Dataset(Dataset):
    def __init__(
        self,
        data_folder: str,
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
        build_image_cache: bool = True,  # Default True to preserve existing behavior
    ):
        """
        Main ViNT dataset class

        Args:
            data_folder (string): Directory with all the image data
            dataset_name (string): Name of the dataset [recon, go_stanford, scand, tartandrive, etc.]
            waypoint_spacing (int): Spacing between waypoints
            min_dist_cat (int): Minimum distance category to use
            max_dist_cat (int): Maximum distance category to use
            negative_mining (bool): Whether to use negative mining from the ViNG paper (Shah et al.) (https://arxiv.org/abs/2012.09812)
            len_traj_pred (int): Length of trajectory of waypoints to predict if this is an action dataset
            learn_angle (bool): Whether to learn the yaw of the robot at each predicted waypoint if this is an action dataset
            context_size (int): Number of previous observations to use as context
            end_slack (int): Number of timesteps to ignore at the end of the trajectory
            goals_per_obs (int): Number of goals to sample per observation
            normalize (bool): Whether to normalize the distances or actions
        """
        self.data_folder = data_folder
        self.split = split
        self.split_ratio = split_ratio
        self.dataset_name = dataset_name
        self.dataset_index = dataset_index

        traj_names_file = os.path.join(self.data_folder, "dataset_metadata.json")
        with open(traj_names_file, "r") as f:
            json_data = json.load(f)
        trajectories = json_data["trajectories"]
        split_point = int(len(trajectories) * split_ratio)
        trajectories = trajectories[:split_point] if split == "train" else trajectories[split_point:]
        self.traj_names = [os.path.join(self.data_folder, traj["path"]) for traj in trajectories]

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
        self.build_image_cache = build_image_cache  # Store the parameter

        self.trajectory_cache = {}
        self._load_index()
        # Only build image cache if requested (default True for backward compatibility)
        if self.build_image_cache:
            self._build_caches()
        else:
            # Set image cache to None when not building it
            self._image_cache = None

        if self.learn_angle:
            self.num_action_params = 3
        else:
            self.num_action_params = 2

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        # Handle backward compatibility: if build_image_cache doesn't exist, default to True
        if not hasattr(self, 'build_image_cache'):
            self.build_image_cache = True
        # Only build image cache if requested
        if self.build_image_cache:
            self._build_caches()
        else:
            self._image_cache = None

    def _build_caches(self, use_tqdm: bool = True):
        """
        Build a cache of images for faster loading using LMDB
        """
        cache_filename = os.path.join(
            self.data_folder,
            f"dataset_{self.dataset_name}_{self.split}_{self.split_ratio}.lmdb",
        )

        # Load all the trajectories into memory. These should already be loaded, but just in case.
        for traj_name in self.traj_names:
            self._get_trajectory(traj_name)

        """
        If the cache file doesn't exist, create it by iterating through the dataset and writing each image to the cache
        """
        if not os.path.exists(cache_filename):
            tqdm_iterator = tqdm.tqdm(
                self.goals_index,
                disable=not use_tqdm,
                dynamic_ncols=True,
                desc=f"Building LMDB cache for {self.dataset_name}",
            )
            with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    for traj_name, time in tqdm_iterator:
                        image_path = get_data_path(self.data_folder, traj_name, time)
                        try:
                            with open(image_path, "rb") as f:
                                txn.put(image_path.encode(), f.read())
                        except Exception as e:
                            print(f"Failed to open {image_path}: {e}")


        # Reopen the cache file in read-only mode
        self._image_cache: lmdb.Environment = lmdb.open(cache_filename, readonly=True)

    def _build_index(self, use_tqdm: bool = False):
        """
        Build an index consisting of tuples (trajectory name, time, max goal distance).
        """
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(
            self.traj_names, disable=not use_tqdm, dynamic_ncols=True
        ):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])

            # Create the goals index
            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size * self.waypoint_spacing
            end_time = (
                traj_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing
            )

            # Create the samples index
            for curr_time in range(begin_time, end_time):
                max_goal_distance = min(
                    self.max_dist_cat * self.waypoint_spacing, traj_len - curr_time - 1
                )
                samples_index.append((traj_name, curr_time, max_goal_distance))

        return samples_index, goals_index

    def _sample_goal(self, trajectory_name, curr_time, max_goal_dist):
        """
        Sample a goal from the future in the same trajectory.
        Returns: (trajectory_name, goal_time, goal_is_negative)
        """
        goal_offset = np.random.randint(0, max_goal_dist + 1)
        if goal_offset == 0:
            trajectory_name, goal_time = self._sample_negative()
            return trajectory_name, goal_time, True
        else:
            goal_time = curr_time + int(goal_offset * self.waypoint_spacing)
            return trajectory_name, goal_time, False

    def _sample_negative(self):
        """
        Sample a goal from a (likely) different trajectory.
        """
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def _load_index(self) -> None:
        """
        Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
        """
        index_to_data_path = os.path.join(
            self.data_folder,
            f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_context_n{self.context_size}_slack_{self.end_slack}.pkl",
        )
        try:
            # load the index_to_data if it already exists (to save time)
            with open(index_to_data_path, "rb") as f:
                self.index_to_data, self.goals_index = pickle.load(f)
        except:
            # if the index_to_data file doesn't exist, create it
            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _load_image(self, trajectory_name, time, retries=3):
        image_path = get_data_path(self.data_folder, trajectory_name, time)

        # If image cache is disabled, load directly from filesystem
        if self._image_cache is None:
            try:
                return img_path_to_data(image_path, self.image_size)
            except Exception as e:
                print(f"Failed to load image {image_path} from filesystem: {e}")
                return None

        # Original LMDB cache loading logic
        attempt = 0
        while attempt < retries:
            try:
                with self._image_cache.begin() as txn:
                    image_buffer = txn.get(image_path.encode())
                    if image_buffer is None:
                        raise TypeError("Key not found in LMDB cache")
                    image_bytes = bytes(image_buffer)
                image_bytes = io.BytesIO(image_bytes)
                return img_path_to_data(image_bytes, self.image_size)
            except TypeError:
                attempt += 1
                print(f"Failed to load image {image_path}, attempt {attempt}/{retries}")
        return None

    def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        yaw = traj_data["yaw"][start_index : end_index : self.waypoint_spacing]
        positions = traj_data["position"][
            start_index : end_index : self.waypoint_spacing
        ]

        goal_pos = np.array(traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)])

        yaw = np.array(yaw) # Using np for avoiding list errors
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

        waypoints = to_local_coords(positions, positions[0], yaw[0])
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
            actions[:, :2] /= (
                self.metric_waypoint_spacing * self.waypoint_spacing
            )
            goal_pos /= (
                self.metric_waypoint_spacing * self.waypoint_spacing
            )

        assert actions.shape == (
            self.len_traj_pred,
            self.num_action_params,
        ), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"

        return actions, goal_pos

    def _get_trajectory(self, trajectory_name):
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]
        else:
            # trajectory_name already contains the full path from data_folder
            traj_data_path = os.path.join(trajectory_name, "traj_data.json")
            with open(traj_data_path, "rb") as f:
                traj_data = json.load(f)
            self.trajectory_cache[trajectory_name] = traj_data
            return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data)

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
        f_curr, curr_time, max_goal_dist = self.index_to_data[i]
        f_goal, goal_time, goal_is_negative = self._sample_goal(
            f_curr, curr_time, max_goal_dist
        )

        # Load images
        context = []
        # sample the last self.context_size times from interval [0, curr_time)
        context_times = list(
            range(
                curr_time + -self.context_size * self.waypoint_spacing,
                curr_time + 1,
                self.waypoint_spacing,
            )
        )
        context = [(f_curr, t) for t in context_times] # context is 4

        obs_image = torch.cat([self._load_image(f, t) for f, t in context]) # obs after concat is (channel * context, H, W) - ( 12, 240, 320)

        # Load goal image
        goal_image = self._load_image(f_goal, goal_time) # goal_image is (3, 240, 320)

        # Load other trajectory data
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        goal_traj_data = self._get_trajectory(f_goal)
        goal_traj_len = len(goal_traj_data["position"])
        assert goal_time < goal_traj_len, f"{goal_time} an {goal_traj_len}"

        # Compute actions
        actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)

        # Compute distances
        if goal_is_negative:
            distance = self.max_dist_cat
        else:
            distance = (goal_time - curr_time) // self.waypoint_spacing
            assert (
                goal_time - curr_time
            ) % self.waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}"

        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        if self.learn_angle:
            actions_torch = calculate_sin_cos(actions_torch)

        action_mask = (
            (distance < self.max_action_distance)
            and (distance > self.min_action_distance)
            and (not goal_is_negative)
        )

        return (
            torch.as_tensor(obs_image, dtype=torch.float32),
            torch.as_tensor(goal_image, dtype=torch.float32),
            actions_torch,
            torch.as_tensor(distance, dtype=torch.int64),
            torch.as_tensor(goal_pos, dtype=torch.float32),
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
            torch.as_tensor(action_mask, dtype=torch.float32),
        )
