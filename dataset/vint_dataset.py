from typing import Tuple
import torch
import tqdm
import os
import json

from .base_dataset import BaseViNTDataset
from .data_utils import find_max_goal_distance_meters


class ViNT_Dataset(BaseViNTDataset):
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
        Main ViNT dataset class

        Args:
            data_folder (string): Directory with all the image data
            split (string): Dataset split ('train' or 'test')
            split_ratio (float): Ratio of data to use for training
            dataset_name (string): Name of the dataset [recon, go_stanford, scand, tartandrive, etc.]
            dataset_index (int): Index of the dataset
            image_size (tuple): Size of the images (width, height)
            waypoint_spacing (int): Spacing between waypoints
            min_goal_distance_meters (float): Minimum goal distance in meters
            max_goal_distance_meters (float): Maximum goal distance in meters
            negative_mining (bool): Whether to use negative mining from the ViNG paper (Shah et al.) (https://arxiv.org/abs/2012.09812)
            len_traj_pred (int): Length of trajectory of waypoints to predict if this is an action dataset
            context_size (int): Number of previous observations to use as context
            end_slack (int): Number of timesteps to ignore at the end of the trajectory
            normalize (bool): Whether to normalize the distances or actions
            force_rebuild_indices (bool): Whether to force rebuild dataset indices
        """
        
        # Load trajectory names from metadata
        traj_names_file = os.path.join(data_folder, "dataset_metadata.json")
        with open(traj_names_file, "r") as f:
            json_data = json.load(f)
        trajectories = json_data["trajectories"]
        if split != "train":
            split_ratio = 1.0 - split_ratio
        split_point = int(len(trajectories) * split_ratio)
        trajectories = trajectories[:split_point] if split == "train" else trajectories[split_point:]
        self.traj_names = [os.path.join(data_folder, traj["path"]) for traj in trajectories]
        
        super().__init__(
            data_folder=data_folder,
            split=split,
            split_ratio=split_ratio,
            dataset_name=dataset_name,
            dataset_index=dataset_index,
            image_size=image_size,
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
                # Use consistent frame-based calculation to avoid index out of bounds
                # Calculate max goal distance in frames, ensuring it doesn't exceed trajectory bounds
                # purely meter‐based max‐goal (in frames)
                max_m = self.max_goal_distance_meters
                max_goal = find_max_goal_distance_meters(traj_data, curr_time, max_m)
                samples_index.append((traj_name, curr_time, max_goal))

        return samples_index, goals_index

        
    def _load_samples(self, trajectory_name, time, goal_trajectory_name, goal_time):
        context_times = self._get_context_times(time)
        context_images = [self._load_image(trajectory_name, t) for t in context_times]
        obs_images = torch.cat(context_images)
        goal_image = self._load_image(goal_trajectory_name, goal_time)
        return obs_images, goal_image
    
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
