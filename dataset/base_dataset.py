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

from .data_utils import (
    to_local_coords,
    calculate_distance_meters,
    calculate_sin_cos,
    get_data_path,
    img_path_to_data,
)

# Import CLI formatter for enhanced error messages
try:
    from visualnav_transformer.train.vint_train.logging.cli_formatter import print_error, Symbols
except ImportError:
    # Fallback if cli_formatter is not available
    def print_error(msg, symbol=None):
        print(f"ERROR: {msg}")
    class Symbols:
        ERROR = '❌'


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
        Initialize the base dataset with common parameters.

        Args:
            data_folder: Path to the data folder
            split: 'train' or 'test'
            split_ratio: Ratio of data to use for training
            dataset_name: Name of the dataset
            dataset_index: Index of the dataset
            image_size: Size of the images (width, height)
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
        super().__init__()
        self.data_folder = data_folder
        self.split = split
        self.split_ratio = split_ratio
        self.dataset_name = dataset_name
        self.dataset_index = dataset_index
        self.image_size = image_size
        self.waypoint_spacing = waypoint_spacing
        self.min_goal_distance_meters = min_goal_distance_meters
        self.max_goal_distance_meters = max_goal_distance_meters
        self.negative_mining = negative_mining
        self.len_traj_pred = len_traj_pred
        self.context_size = context_size
        self.end_slack = end_slack
        self.normalize = normalize
        self.force_rebuild_indices = force_rebuild_indices
        self.num_action_params = 3 # x, y, yaw
        self._load_index(force_rebuild=force_rebuild_indices)
    
    def _load_index(self, force_rebuild=False) -> None:
        """
        Load or build the dataset index with pickle caching.
        """
        index_to_data_path = os.path.join(
            self.data_folder,
            f"dataset_dist_{self.max_goal_distance_meters}.pkl",
        )
        try:
            if force_rebuild:
                raise FileNotFoundError
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
        min_goal_dist = 0 if self.negative_mining else 1
        if max_goal_dist <= 0:
            raise ValueError(f"Warning: max_goal_dist is {max_goal_dist} for {trajectory_name} at time {curr_time}")
        goal_offset = np.random.randint(min_goal_dist, max_goal_dist + 1)
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
        # Load trajectory data from JSON file
        traj_data_path = os.path.join(trajectory_name, "traj_data.json")
        with open(traj_data_path, "r") as f:
            traj_data = json.load(f)
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
        
        goal_yaw = yaw[-1] - yaw[0]
        # convert to sin(yaw), cos(yaw)
        goal_yaw = np.array([np.sin(goal_yaw), np.cos(goal_yaw)])
        goal_pos = np.concatenate([goal_pos, goal_yaw])

        assert actions.shape == (
            self.len_traj_pred,
            self.num_action_params,
        ), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"
        
        if goal_is_negative:
            # TODO: set maximum frame distance (compute from frame rate statistics)
            distance = self.max_goal_distance_meters  # Use max distance for negatives
        else:
            distance = (goal_time - curr_time) // self.waypoint_spacing
            
        action_mask = (
            (distance < self.max_goal_distance_meters)
            and (distance > self.min_goal_distance_meters)
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
            print_error(f"Failed to load image {image_path}: {e}", Symbols.ERROR)
            return None
