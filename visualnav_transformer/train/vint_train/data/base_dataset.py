"""
Base dataset class for ViNT datasets.

This module contains the BaseViNTDataset class that provides shared functionality
between ViNT_Dataset and FeatureDataset classes.
"""

import json
import os
import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from visualnav_transformer.train.vint_train.data.data_utils import (
    to_local_coords,
    calculate_distance_meters,
    calculate_sin_cos,
    get_data_path,
    img_path_to_data,
)


class BaseViNTDataset(Dataset, ABC):
    """
    Base class for ViNT datasets that provides shared functionality.
    
    This class contains all the common logic between ViNT_Dataset and FeatureDataset,
    including initialization, goal sampling, action computation, and trajectory management.
    """
    
    def __init__(
        self,
        data_folder: str,
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
        Initialize the base dataset with common parameters.
        
        Args:
            data_folder: Path to the data folder
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
            **kwargs: Additional parameters for subclasses
        """
        super().__init__()
        
        # Store all common parameters
        self.data_folder = data_folder
        self.split = split
        self.split_ratio = split_ratio
        self.dataset_name = dataset_name
        self.dataset_index = dataset_index
        self.image_size = image_size
        self.waypoint_spacing = waypoint_spacing
        self.min_goal_distance_meteres = min_goal_distance_meteres
        self.max_goal_distance_meters = max_goal_distance_meters
        self.negative_mining = negative_mining
        self.len_traj_pred = len_traj_pred
        self.context_size = context_size
        self.end_slack = end_slack
        self.goals_per_obs = goals_per_obs
        self.normalize = normalize
        
        # Load trajectory names from metadata
        traj_names_file = os.path.join(self.data_folder, "dataset_metadata.json")
        with open(traj_names_file, "r") as f:
            json_data = json.load(f)
        trajectories = json_data["trajectories"]
        split_point = int(len(trajectories) * split_ratio)
        trajectories = trajectories[:split_point] if split == "train" else trajectories[split_point:]
        self.traj_names = [os.path.join(self.data_folder, traj["path"]) for traj in trajectories]
        
        self.num_action_params = 3 # x, y, yaw
        self.trajectory_cache = {}
        # Load the index
        self._load_index()
    
    def _load_index(self) -> None:
        """
        Load or build the dataset index with pickle caching.
        """
        index_to_data_path = os.path.join(
            self.data_folder,
            f"dataset_dist_{self.max_goal_distance_meters}_context_n{self.context_size}_slack_{self.end_slack}.pkl",
        )
        try:
            # Load the index if it already exists
            with open(index_to_data_path, "rb") as f:
                self.index_to_data, self.goals_index = pickle.load(f)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            # Build the index if it doesn't exist or is corrupted
            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)
    
    @abstractmethod
    def _build_index(self) -> Tuple[List, List]:
        """
        Build the dataset index.
        
        Returns:
            Tuple of (index_to_data, goals_index)
        """
        pass
    
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
        
        Returns:
            Tuple of (trajectory_name, goal_time)
        """
        return self.goals_index[np.random.randint(0, len(self.goals_index))]
    
    def _get_trajectory(self, trajectory_name):
        """
        Get trajectory data for a given trajectory name with caching.

        Args:
            trajectory_name: Name of the trajectory

        Returns:
            Dictionary containing trajectory data
        """
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]
        else:
            # Load trajectory data from JSON file
            traj_data_path = os.path.join(trajectory_name, "traj_data.json")
            with open(traj_data_path, "r") as f:
                traj_data = json.load(f)
            self.trajectory_cache[trajectory_name] = traj_data
            return traj_data

    def _compute_actions(self, traj_data, curr_time, goal_time, goal_is_negative):
        """
        Compute actions and goal position using waypoint trajectories.

        Args:
            traj_data: Trajectory data dictionary
            curr_time: Current time step
            goal_time: Goal time step
            goal_is_negative: Whether this is a negative sample

        Returns:
            Tuple of (actions, goal_pos, distance, action_mask)
        """
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        yaw = np.array(traj_data["yaw"][start_index : end_index : self.waypoint_spacing])
        positions = np.array(traj_data["position"][start_index : end_index : self.waypoint_spacing])
        goal_pos = np.array(traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)])

        # YAW       
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

        # POSITIONS
        waypoints = to_local_coords(positions, positions[0], yaw[0])
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])

        assert waypoints.shape == (
            self.len_traj_pred + 1,
            2,
        ), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        yaw = yaw[1:] - yaw[0]
        actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)

        if self.normalize:
            actions[:, :2] /= self.max_goal_distance_meters
            goal_pos /= self.max_goal_distance_meters

        assert actions.shape == (
            self.len_traj_pred,
            self.num_action_params,
        ), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"
        
        if goal_is_negative:
            distance = self.max_goal_distance_meters  # Use max distance for negatives
        else:
            distance = calculate_distance_meters(traj_data["position"][curr_time], traj_data["position"][goal_time])
            
        action_mask = (
            (distance < self.max_goal_distance_meters)
            and (distance > self.min_goal_distance_meteres)
            and (not goal_is_negative)
        )
        
        return actions, goal_pos, distance, action_mask


    def _get_context_times(self, curr_time):
        """
        Get the time steps for context frames.

        Args:
            curr_time: Current time step

        Returns:
            List of time steps for context frames
        """
        return list(
            range(
                curr_time + -self.context_size * self.waypoint_spacing,
                curr_time + 1,
                self.waypoint_spacing,
            )
        )

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            Number of samples in the dataset
        """
        return len(self.index_to_data)
    
    def _load_image(self, trajectory_name, time):
        image_path = get_data_path(self.data_folder, trajectory_name, time)
        try:
            return img_path_to_data(image_path, self.image_size)
        except Exception as e:
            print(f"Failed to load image {image_path} from filesystem: {e}")
            return None

    @abstractmethod
    def _load_samples(self, trajectory_name, time, goal_trajectory_name, goal_time):
        pass
    
    
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

        # Load observation images, goal_images
        obs_image, goal_image = self._load_samples(f_curr, curr_time, f_goal, goal_time)

        # Load other trajectory data
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
            torch.as_tensor(obs_image, dtype=torch.float32),
            torch.as_tensor(goal_image, dtype=torch.float32),
            actions_torch,
            torch.as_tensor(distance, dtype=torch.int64),
            torch.as_tensor(goal_pos, dtype=torch.float32),
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
            torch.as_tensor(action_mask, dtype=torch.float32),
        )
