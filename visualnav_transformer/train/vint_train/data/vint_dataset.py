from typing import Tuple
import torch
import tqdm
import os
import json

from visualnav_transformer.train.vint_train.data.base_dataset import BaseViNTDataset
from visualnav_transformer.train.vint_train.data.data_utils import find_max_goal_distance_meters


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
        min_goal_distance_meteres: float,
        max_goal_distance_meters: float,
        negative_mining: bool,
        len_traj_pred: int,
        learn_angle: bool,
        context_size: int,
        end_slack: int = 0,
        goals_per_obs: int = 1,
        normalize: bool = True,
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
            build_image_cache (bool): Whether to build LMDB image cache
        """
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
            learn_angle=learn_angle,
            context_size=context_size,
            end_slack=end_slack,
            goals_per_obs=goals_per_obs,
            normalize=normalize,
        )
        
        # Load trajectory names from metadata
        traj_names_file = os.path.join(self.data_folder, "dataset_metadata.json")
        with open(traj_names_file, "r") as f:
            json_data = json.load(f)
        trajectories = json_data["trajectories"]
        split_point = int(len(trajectories) * split_ratio)
        trajectories = trajectories[:split_point] if split == "train" else trajectories[split_point:]
        self.traj_names = [os.path.join(self.data_folder, traj["path"]) for traj in trajectories]
            

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
