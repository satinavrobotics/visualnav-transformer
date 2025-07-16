import os
import torch
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset

from .dataset.feature_dataset import FeatureDataset

class GoalModuleDataset(FeatureDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
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
        super:
            torch.as_tensor(obs_features, dtype=torch.float32),
            torch.as_tensor(goal_feature, dtype=torch.float32),
            actions_torch,
            torch.as_tensor(distance, dtype=torch.int64),
            torch.as_tensor(goal_pos, dtype=torch.float32),
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
            torch.as_tensor(action_mask, dtype=torch.float32),
        """

            
        obs_features, goal_feature, _, _, goal_pos, _, _ = super().__getitem__(i)
        features = torch.cat([obs_features[-1].unsqueeze(0), goal_feature.unsqueeze(0)], dim=0)
        return (
            features,   # [2, 1024]
            goal_pos,   # [4]
        )
    
class DummyGoalDataset(Dataset):
    """
    Dummy dataset for testing the pipeline with goal-oriented data structure.

    Returns:
        - features: [N_OBSERVATIONS, 1024] tensor
        - goal_features: [1, 1024] tensor
        - actions: [8, 3] tensor
        - goal_position: [3] tensor containing goal_x, goal_y, goal_yaw
        - dataset_index: [1] tensor
    """

    def __init__(self, train=True, n_samples=1000):
        self.train = train
        self.n_samples = n_samples

        # Generate dummy data
        self._generate_dummy_data()

    def _generate_dummy_data(self):
        """Generate dummy data with the specified shapes."""
        # Set random seed for reproducibility
        torch.manual_seed(42 if self.train else 123)
        np.random.seed(42 if self.train else 123)

        # Generate features for each sample [N_SAMPLES, N_OBSERVATIONS, 1024]
        # 2 is a hardcoded value, just random
        self.features = torch.randn(self.n_samples, 2, 1024)
        # Generate goal features for each sample [N_SAMPLES, 1, 1024]
        self.goal_features = torch.randn(self.n_samples, 1, 1024)
        # Generate goal positions [N_SAMPLES, 3] (goal_x, goal_y, goal_yaw)
        self.goal_positions = torch.randn(self.n_samples, 3)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (features, goal_features, actions, goal_position, dataset_index)
        """
        current_features = self.features[index][0]
        goal_features = self.goal_features[index][0]
        features = torch.cat([current_features, goal_features], dim=0)
        
        return (
            features,   # [N_OBSERVATIONS, 1024]
            self.goal_positions[index],             # [3]
        )

    def __len__(self):
        return self.n_samples