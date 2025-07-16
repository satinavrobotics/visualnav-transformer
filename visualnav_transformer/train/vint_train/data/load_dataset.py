import os

from torch.utils.data import ConcatDataset, DataLoader

from visualnav_transformer.train.vint_train.data.dataset.data_utils import infinite_loader
from visualnav_transformer.train.vint_train.data.dataset.vint_dataset import ViNT_Dataset
from visualnav_transformer.train.vint_train.data.dataset.feature_dataset import FeatureDataset
from visualnav_transformer.train.vint_train.data.viz_hybrid_dataset import VizHybridDataset
from visualnav_transformer.train.vint_train.logging.cli_formatter import (
    Colors, Symbols, print_info, print_success, print_warning, print_error,
    print_section, format_number
)

def load_dataset(config, data_configs):
    """Load training and test datasets with enhanced CLI output."""
    print_section("Loading Datasets", Symbols.LOAD, Colors.BRIGHT_CYAN)

    # Load the data
    train_dataset = []
    test_dataloaders = {}
    total_datasets = len([name for name, cfg in data_configs["datasets"].items()
                         if cfg.get("available", False) and name != "viz_data"])

    print_info(f"Found {total_datasets} available datasets", Symbols.INFO)

    for dataset_index, dataset_name in enumerate(data_configs["datasets"]):
        data_config = data_configs["datasets"][dataset_name]
        if not bool(data_config["available"]):
            print_warning(f"Dataset {dataset_name} is not available - skipping", Symbols.WARNING)
            continue

        # Skip viz_data dataset - it's only used for visualization, not training
        if dataset_name == "viz_data":
            print_info(f"Skipping {dataset_name} - visualization dataset only", Symbols.INFO)
            continue
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        train_split = data_config["split"]
        test_split = 1 - train_split

        # Display dataset configuration
        print_info(f"Processing dataset: {Colors.BRIGHT_YELLOW}{dataset_name}{Colors.RESET}", Symbols.BULLET)
        print(f"  {Colors.CYAN}Data folder:{Colors.RESET} {data_config['data_folder']}")
        print(f"  {Colors.CYAN}Train/Test split:{Colors.RESET} {train_split:.1%}/{test_split:.1%}")
        print(f"  {Colors.CYAN}Waypoint spacing:{Colors.RESET} {data_config.get('waypoint_spacing', 1)}")
        print(f"  {Colors.CYAN}Negative mining:{Colors.RESET} {data_config.get('negative_mining', False)}")

        if train_split != 0.0:
            # Check if using pre-built DINO features
            use_prebuilt_features = config.get("prebuilt_dino", False)

            if use_prebuilt_features:
                print(f"  {Colors.MAGENTA}Using pre-built DINO features{Colors.RESET}")
            else:
                print(f"  {Colors.MAGENTA}Using raw images with DINO encoder{Colors.RESET}")

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
                    min_goal_distance_meters=data_configs["min_goal_distance_meters"],
                    max_goal_distance_meters=data_configs["max_goal_distance_meters"],
                    negative_mining=data_config["negative_mining"],
                    len_traj_pred=config["len_traj_pred"],
                    context_size=config["context_size"],
                    end_slack=data_config["end_slack"],
                    normalize=config["normalize"],
                    force_rebuild_indices=data_configs.get("force_rebuild_indices", False)
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
                    waypoint_spacing=data_config["waypoint_spacing"],
                    min_goal_distance_meters=data_configs["min_goal_distance_meters"],
                    max_goal_distance_meters=data_configs["max_goal_distance_meters"],
                    negative_mining=data_config["negative_mining"],
                    len_traj_pred=config["len_traj_pred"],
                    context_size=config["context_size"],
                    end_slack=data_config["end_slack"],
                    normalize=config["normalize"],
                    force_rebuild_indices=data_configs.get("force_rebuild_indices", False)
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
                    min_goal_distance_meters=data_configs["min_goal_distance_meters"],
                    max_goal_distance_meters=data_configs["max_goal_distance_meters"],
                    negative_mining=data_config["negative_mining"],
                    len_traj_pred=config["len_traj_pred"],
                    context_size=config["context_size"],
                    end_slack=data_config["end_slack"],
                    normalize=config["normalize"],
                    force_rebuild_indices=data_configs.get("force_rebuild_indices", False)
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
                    waypoint_spacing=data_config["waypoint_spacing"],
                    min_goal_distance_meters=data_configs["min_goal_distance_meters"],
                    max_goal_distance_meters=data_configs["max_goal_distance_meters"],
                    negative_mining=data_config["negative_mining"],
                    len_traj_pred=config["len_traj_pred"],
                    context_size=config["context_size"],
                    end_slack=data_config["end_slack"],
                    normalize=config["normalize"],
                    force_rebuild_indices=data_configs.get("force_rebuild_indices", False)
                )

            test_dataloaders[dataset_name] = dataset

        # Enhanced dataset summary
        total_samples = len(dataset) if 'dataset' in locals() else 0
        train_samples = int(total_samples * train_split) if train_split > 0 else 0
        test_samples = int(total_samples * test_split) if test_split > 0 else 0

        print_success(f"Dataset {dataset_name} loaded successfully", Symbols.SUCCESS)
        print(f"  {Colors.GREEN}Total samples:{Colors.RESET} {format_number(total_samples, 0)}")
        if train_samples > 0:
            print(f"  {Colors.BLUE}Training samples:{Colors.RESET} {format_number(train_samples, 0)}")
        if test_samples > 0:
            print(f"  {Colors.MAGENTA}Test samples:{Colors.RESET} {format_number(test_samples, 0)}")
        print()  # Add spacing between datasets

    # Combine all the datasets from different robots
    train_dataset = ConcatDataset(train_dataset)

    print_section("Creating DataLoaders", Symbols.ROCKET, Colors.BRIGHT_BLUE)
    print_info(f"Combined training dataset: {format_number(len(train_dataset), 0)} total samples", Symbols.INFO)
    print_info(f"Training batch size: {config['batch_size']}", Symbols.INFO)
    print_info(f"Number of workers: {config['num_workers']}", Symbols.INFO)

    # Reduce memory usage for DataLoader to prevent worker crashes
    # If using pre-built DINO features, use minimal workers and disable persistent workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=config["num_workers"],
        pin_memory=False,  # Disable pin_memory to reduce memory usage
    )

    print_success(f"Training dataloader created: {len(train_loader)} batches", Symbols.SUCCESS)

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    print_info(f"Evaluation batch size: {config['eval_batch_size']}", Symbols.INFO)

    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        print_success(f"Test dataloader '{dataset_type}': {len(test_dataloaders[dataset_type])} batches", Symbols.SUCCESS)

    print_success(f"All datasets loaded successfully! Ready for training.", Symbols.ROCKET)
    print()
    return train_loader, test_dataloaders


def load_viz_dataset(config, data_config):
    """Load visualization datasets with simplified enhanced output."""
    print_section("Loading Visualization Datasets", Symbols.CHART, Colors.BRIGHT_MAGENTA)

    # Use the dedicated visualization dataset
    viz_dataset_name = "viz_data"
    viz_dataset_config = data_config["datasets"].get(viz_dataset_name)

    if viz_dataset_config is None or not viz_dataset_config.get("available", False):
        print_warning("Visualization dataset not found or not available", Symbols.WARNING)
        return None, None

    print_info(f"Using dedicated visualization dataset: {viz_dataset_name}", Symbols.INFO)
    
    train_viz_folder = os.path.join(viz_dataset_config["data_folder"], "train_viz")
    train_dino_cache_folder = os.path.join(train_viz_folder, "dino_cache_large")
    test_viz_folder = os.path.join(viz_dataset_config["data_folder"], "test_viz")
    test_dino_cache_folder = os.path.join(test_viz_folder, "dino_cache_large")

    print_info(f"Train viz folder: {train_viz_folder}", Symbols.LOAD)
    print_info(f"Test viz folder: {test_viz_folder}", Symbols.LOAD)

    if not os.path.exists(train_dino_cache_folder):
        print_warning(f"DINO cache not found at {train_dino_cache_folder}", Symbols.WARNING)
        return None, None
    if not os.path.exists(test_dino_cache_folder):
        print_warning(f"DINO cache not found at {test_dino_cache_folder}", Symbols.WARNING)
        return None, None

    train_viz_dataset = None
    try:
        train_viz_dataset = VizHybridDataset(
            viz_folder=train_viz_folder,
            feature_folder=train_dino_cache_folder,
            context_size=config["context_size"],
            waypoint_spacing=viz_dataset_config.get("waypoint_spacing", 1),
            len_traj_pred=config["len_traj_pred"],
            normalize=config["normalize"],
            split="train",
            split_ratio=1.0,
            dataset_name="viz_hybrid",
            dataset_index=0,
            min_goal_distance_meters=1.0,
            max_goal_distance_meters=10.3,
            negative_mining=False,
            end_slack=0,
            force_rebuild_indices=data_config.get("force_rebuild_indices", False),
        )
        
        # Simplified dataset loading summary
        if train_viz_dataset is not None:
            print_success(f"Train viz dataset: {len(train_viz_dataset)} samples, {len(train_viz_dataset.traj_names)} trajectories", Symbols.SUCCESS)
        else:
            print_error("Train VizHybridDataset not loaded (DINO cache missing)", Symbols.ERROR)
            
        test_viz_dataset = VizHybridDataset(
            viz_folder=test_viz_folder,
            feature_folder=test_dino_cache_folder,
            context_size=config["context_size"],
            waypoint_spacing=viz_dataset_config.get("waypoint_spacing", 1),
            len_traj_pred=config["len_traj_pred"],
            normalize=config["normalize"],
            split="test",
            split_ratio=1.0,
            dataset_name="viz_hybrid",
            dataset_index=0,
            min_goal_distance_meters=1.0,
            max_goal_distance_meters=10.3,
            negative_mining=False,
            end_slack=0,
            force_rebuild_indices=data_config.get("force_rebuild_indices", False),
        )
            
        # Simplified dataset loading summary
        if test_viz_dataset is not None:
            print_success(f"Test viz dataset: {len(test_viz_dataset)} samples, {len(test_viz_dataset.traj_names)} trajectories", Symbols.SUCCESS)
        else:
            print_error("Test VizHybridDataset not loaded (DINO cache missing)", Symbols.ERROR)

    except Exception as e:
        print_error(f"Failed to load visualization dataset: {e}", Symbols.ERROR)
        return None, None
    
    
    # Create dataloaders
    train_viz_dataloader = DataLoader(
        train_viz_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,  # Use 0 workers to avoid issues
        pin_memory=True
    )
    test_viz_dataloader = DataLoader(
        test_viz_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,  # Use 0 workers to avoid issues
        pin_memory=True
    )

    # Wrap so next() never raises StopIteration
    train_viz_dataloader = infinite_loader(train_viz_dataloader)
    test_viz_dataloader = infinite_loader(test_viz_dataloader)

    print_success("Visualization dataloaders created successfully", Symbols.ROCKET)
    print_info(f"Batch size: 32, Infinite loading enabled", Symbols.INFO)
    print()

    return train_viz_dataloader, test_viz_dataloader
