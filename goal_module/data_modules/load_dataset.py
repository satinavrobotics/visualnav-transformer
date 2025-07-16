import os
from torch.utils.data import ConcatDataset

from .goal_dataset import GoalModuleDataset

def load_dataset_train(data_configs):
    # Load the data
    train_dataset = []
    print("Loading train datasets...")
    for dataset_index, dataset_name in enumerate(data_configs["datasets"]):
        data_config = data_configs["datasets"][dataset_name]
        if not bool(data_config["available"]):
            continue
        
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        train_split = data_config["split"]
        feature_folder = os.path.join(data_config["data_folder"], "dino_cache_large")

        if train_split != 0.0:     
            dataset = GoalModuleDataset(
                data_folder=data_config["data_folder"],
                feature_folder=feature_folder,
                split="train",
                split_ratio=train_split,
                dataset_name=dataset_name,
                dataset_index=dataset_index,
                waypoint_spacing=data_config["waypoint_spacing"],
                min_goal_distance_meters=data_configs["min_goal_distance_meters"],
                max_goal_distance_meters=data_configs["max_goal_distance_meters"],
                negative_mining=False,
                len_traj_pred=1,
                context_size=0,
                end_slack=data_config["end_slack"],
                normalize=True
            )
            train_dataset.append(dataset)

    train_dataset = ConcatDataset(train_dataset)
    return train_dataset


def load_dataset_test(data_configs):
    # Load the data
    test_dataset = []
    print("Loading test datasets..")
    for dataset_index, dataset_name in enumerate(data_configs["datasets"]):
        data_config = data_configs["datasets"][dataset_name]
        if not bool(data_config["available"]):
            continue
        
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        test_split = 1 - data_config["split"]
        feature_folder = os.path.join(data_config["data_folder"], "dino_cache_large")

        if test_split != 0.0:
            feature_folder = os.path.join(data_config["data_folder"], "dino_cache_large")
            dataset = GoalModuleDataset(
                data_folder=data_config["data_folder"],
                feature_folder=feature_folder,
                split="test",
                split_ratio=test_split,
                dataset_name=dataset_name,
                dataset_index=dataset_index,
                waypoint_spacing=1,
                min_goal_distance_meters=data_configs["min_goal_distance_meters"],
                max_goal_distance_meters=data_configs["max_goal_distance_meters"],
                negative_mining=False,
                len_traj_pred=1,
                context_size=0,
                end_slack=data_config["end_slack"],
                normalize=True
            )
            test_dataset.append(dataset)

    test_dataset = ConcatDataset(test_dataset)
    return test_dataset
