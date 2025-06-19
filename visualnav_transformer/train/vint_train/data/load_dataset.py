import os

from torch.utils.data import ConcatDataset, DataLoader

from visualnav_transformer.train.vint_train.data.vint_dataset import ViNT_Dataset
from visualnav_transformer.train.vint_train.data.feature_dataset import FeatureDataset

def load_dataset(config, data_configs):
    # Load the data
    train_dataset = []
    test_dataloaders = {}

    for dataset_index, dataset_name in enumerate(data_configs["datasets"]):
        data_config = data_configs["datasets"][dataset_name]
        if not bool(data_config["available"]):
            continue

        # Skip viz_data dataset - it's only used for visualization, not training
        if dataset_name == "viz_data":
            print(f"Skipping {dataset_name} - visualization dataset only")
            continue
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        train_split = data_config["split"]
        test_split = 1 - train_split

        if train_split != 0.0:
            # Check if using pre-built DINO features
            use_prebuilt_features = config.get("prebuilt_dino", False)

            if not use_prebuilt_features:
                # Use the original ViNT_Dataset
                dataset = ViNT_Dataset(
                    data_folder=data_config["data_folder"],
                    split="train",
                    split_ratio=train_split,
                    dataset_name=dataset_name,
                    dataset_index=dataset_index,
                    image_size=config["image_size"],
                    waypoint_spacing=data_config["waypoint_spacing"],
                    metric_waypoint_spacing=data_config["metric_waypoint_spacing"],
                    min_dist_cat=config["distance"]["min_dist_cat"],
                    max_dist_cat=config["distance"]["max_dist_cat"],
                    min_action_distance=config["action"]["min_dist_cat"],
                    max_action_distance=config["action"]["max_dist_cat"],
                    negative_mining=data_config["negative_mining"],
                    len_traj_pred=config["len_traj_pred"],
                    learn_angle=config["learn_angle"],
                    context_size=config["context_size"],
                    end_slack=data_config["end_slack"],
                    goals_per_obs=data_config["goals_per_obs"],
                    normalize=config["normalize"]
                )
            else:
                # Use the FeatureDataset for pre-built DINO features
                # Construct the feature folder path (assuming it's in a dino_cache_large subfolder)
                feature_folder = os.path.join(data_config["data_folder"], "dino_cache_large")

                dataset = FeatureDataset(
                    data_folder=data_config["data_folder"],
                    feature_folder=feature_folder,
                    split="train",
                    split_ratio=train_split,
                    dataset_name=dataset_name,
                    dataset_index=dataset_index,
                    image_size=config["image_size"],
                    waypoint_spacing=data_config["waypoint_spacing"],
                    metric_waypoint_spacing=data_config["metric_waypoint_spacing"],
                    min_dist_cat=config["distance"]["min_dist_cat"],
                    max_dist_cat=config["distance"]["max_dist_cat"],
                    min_action_distance=config["action"]["min_dist_cat"],
                    max_action_distance=config["action"]["max_dist_cat"],
                    negative_mining=data_config["negative_mining"],
                    len_traj_pred=config["len_traj_pred"],
                    learn_angle=config["learn_angle"],
                    context_size=config["context_size"],
                    end_slack=data_config["end_slack"],
                    goals_per_obs=data_config["goals_per_obs"],
                    normalize=config["normalize"]
                )

            train_dataset.append(dataset)

        if test_split != 0.0:
            # Check if using pre-built DINO features
            use_prebuilt_features = config.get("prebuilt_dino", False)

            if not use_prebuilt_features:
                # Use the original ViNT_Dataset
                dataset = ViNT_Dataset(
                    data_folder=data_config["data_folder"],
                    split="test",
                    split_ratio=test_split,
                    dataset_name=dataset_name,
                    dataset_index=dataset_index,
                    image_size=config["image_size"],
                    waypoint_spacing=data_config["waypoint_spacing"],
                    metric_waypoint_spacing=data_config["metric_waypoint_spacing"],
                    min_dist_cat=config["distance"]["min_dist_cat"],
                    max_dist_cat=config["distance"]["max_dist_cat"],
                    min_action_distance=config["action"]["min_dist_cat"],
                    max_action_distance=config["action"]["max_dist_cat"],
                    negative_mining=data_config["negative_mining"],
                    len_traj_pred=config["len_traj_pred"],
                    learn_angle=config["learn_angle"],
                    context_size=config["context_size"],
                    end_slack=data_config["end_slack"],
                    goals_per_obs=data_config["goals_per_obs"],
                    normalize=config["normalize"]
                )
            else:
                # Use the FeatureDataset for pre-built DINO features
                # Construct the feature folder path (assuming it's in a dino_cache_large subfolder)
                feature_folder = os.path.join(data_config["data_folder"], "dino_cache_large")

                dataset = FeatureDataset(
                    data_folder=data_config["data_folder"],
                    feature_folder=feature_folder,
                    split="test",
                    split_ratio=test_split,
                    dataset_name=dataset_name,
                    dataset_index=dataset_index,
                    image_size=config["image_size"],
                    waypoint_spacing=data_config["waypoint_spacing"],
                    metric_waypoint_spacing=data_config["metric_waypoint_spacing"],
                    min_dist_cat=config["distance"]["min_dist_cat"],
                    max_dist_cat=config["distance"]["max_dist_cat"],
                    min_action_distance=config["action"]["min_dist_cat"],
                    max_action_distance=config["action"]["max_dist_cat"],
                    negative_mining=data_config["negative_mining"],
                    len_traj_pred=config["len_traj_pred"],
                    learn_angle=config["learn_angle"],
                    context_size=config["context_size"],
                    end_slack=data_config["end_slack"],
                    goals_per_obs=data_config["goals_per_obs"],
                    normalize=config["normalize"]
                )

            # Use dataset_name as the dataset_type if not explicitly defined
            dataset_type = dataset_name
            test_dataloaders[dataset_type] = dataset

    # combine all the datasets from different robots
    train_dataset = ConcatDataset(train_dataset)

    # Reduce memory usage for DataLoader to prevent worker crashes
    # If using pre-built DINO features, use minimal workers and disable persistent workers
    use_prebuilt_features = config.get("prebuilt_dino", False)
    if use_prebuilt_features:
        print("Using minimal DataLoader settings for pre-built DINO features to prevent memory issues")
        num_workers = 0  # Use no workers (single-process loading)
        persistent_workers = False  # Disable persistent workers
        batch_size = min(64, config["batch_size"])  # Reduce batch size if needed
        print(f"Using batch size {batch_size} and {num_workers} workers")
    else:
        num_workers = config["num_workers"]
        persistent_workers = True
        batch_size = config["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=persistent_workers,
        pin_memory=False,  # Disable pin_memory to reduce memory usage
    )

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type, dataset in test_dataloaders.items():
        # Use the same minimal settings for test dataloaders when using pre-built DINO features
        if use_prebuilt_features:
            test_batch_size = min(64, config["eval_batch_size"])
            test_num_workers = 0
        else:
            test_batch_size = config["eval_batch_size"]
            test_num_workers = 0  # Already set to 0

        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=test_batch_size,
            shuffle=True,
            num_workers=test_num_workers,
            drop_last=False,
        )
        
    return train_loader, test_dataloaders
