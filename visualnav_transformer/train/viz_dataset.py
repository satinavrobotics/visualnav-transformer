#!/usr/bin/env python3
"""
Visualization Cache Builder
(visualization_cache_builder.py)

This script builds a unified visualization cache for datasets that have camera metrics.
It selects trajectories that meet length criteria and duplicates them into a
single 'Viz_data' folder in the main Sati_data directory.

The script creates train_viz and/or test_viz folders based on the split ratios
of the source datasets:
- If split = 1.0: Only train_viz is created
- If split = 0.0: Only test_viz is created
- If 0.0 < split < 1.0: Both train_viz and test_viz are created

Each viz folder contains:
1. lmdb folder with cache
2. metadata JSON file
3. dataset folders (e.g., recon) with duplicated trajectories directly in the viz folder
"""

import os
import json
import yaml
import random
import shutil
import argparse
import lmdb
import pickle
import numpy as np
from tqdm import tqdm
import cv2

def build_visualization_cache(num_train_trajectories=1, num_test_trajectories=5, min_length=10, max_length=100):
    """
    Build unified visualization cache for datasets with camera metrics.

    Args:
        num_train_trajectories: Number of trajectories to sample per dataset for train split
        num_test_trajectories: Number of trajectories to sample per dataset for test split
        min_length: Minimum trajectory length
        max_length: Maximum trajectory length
    """
    # Load data config
    with open("vint_train/data/data_config.yaml", "r") as f:
        data_configs = yaml.safe_load(f)["datasets"]

    print(f"Building unified visualization cache:")
    print(f"  - Train: {num_train_trajectories} trajectories per dataset")
    print(f"  - Test: {num_test_trajectories} trajectories per dataset")
    print(f"  - Trajectory length criteria: {min_length} to {max_length} frames")

    # Find the Sati_data root directory (parent of all dataset folders)
    sati_data_root = None
    for dataset_name, config in data_configs.items():
        if config.get("available", False):
            data_folder = config["data_folder"]
            # Extract the Sati_data root directory
            sati_data_root = os.path.dirname(data_folder)
            break

    if sati_data_root is None:
        print("Error: Could not find Sati_data root directory")
        return

    # Create unified visualization dataset folder
    viz_data_dir = os.path.join(sati_data_root, "Viz_data")
    os.makedirs(viz_data_dir, exist_ok=True)

    # Create train and test visualization folders
    train_viz_dir = os.path.join(viz_data_dir, "train_viz")
    train_traj_dir = train_viz_dir  # Place trajectories directly in train_viz folder
    train_lmdb_dir = os.path.join(train_viz_dir, "lmdb")
    train_metadata_path = os.path.join(train_viz_dir, "dataset_metadata.json")

    test_viz_dir = os.path.join(viz_data_dir, "test_viz")
    test_traj_dir = test_viz_dir  # Place trajectories directly in test_viz folder
    test_lmdb_dir = os.path.join(test_viz_dir, "lmdb")
    test_metadata_path = os.path.join(test_viz_dir, "dataset_metadata.json")

    # Check if visualization cache already exists
    if os.path.exists(train_metadata_path) and os.path.exists(test_metadata_path):
        print(f"Unified visualization cache already exists at {viz_data_dir}, skipping")
        return

    # Create directories
    os.makedirs(train_viz_dir, exist_ok=True)
    os.makedirs(train_lmdb_dir, exist_ok=True)
    os.makedirs(test_viz_dir, exist_ok=True)
    os.makedirs(test_lmdb_dir, exist_ok=True)

    # Collect trajectories from all datasets
    train_trajectories_by_dataset = {}
    test_trajectories_by_dataset = {}

    # Process each dataset
    for dataset_name, config in data_configs.items():
        print(f"\n{'='*50}")
        print(f"Checking dataset: {dataset_name}")
        print(f"{'='*50}")

        # Skip datasets that aren't available
        if not config.get("available", False):
            print(f"❌ Skipping {dataset_name} - not available")
            continue

        # Skip datasets without camera metrics
        if not has_camera_metrics(config):
            print(f"❌ Skipping {dataset_name} - no camera metrics")
            print(f"   Camera metrics present: {'camera_metrics' in config}")
            if 'camera_metrics' in config:
                print(f"   Camera matrix present: {'camera_matrix' in config['camera_metrics']}")
            continue

        # Get dataset folder
        data_folder = config["data_folder"]
        print(f"✅ Processing dataset: {dataset_name}")
        print(f"   Data folder: {data_folder}")
        print(f"   Has camera metrics: ✅")

        # Determine split ratio
        split_ratio = config.get("split", 0.9)
        print(f"   Split ratio: {split_ratio}")

        # Load dataset metadata
        metadata_path = os.path.join(data_folder, "dataset_metadata.json")
        if not os.path.exists(metadata_path):
            print(f"❌ Metadata not found for {dataset_name} at {metadata_path}")
            continue

        print(f"   Metadata file: ✅")

        with open(metadata_path, "r") as f:
            dataset_metadata = json.load(f)

        # Get all trajectories
        trajectories = dataset_metadata.get("trajectories", [])
        print(f"   Total trajectories: {len(trajectories)}")

        # Process train split if applicable
        if split_ratio > 0.0:
            print(f"Processing {dataset_name} train split (ratio: {split_ratio})")

            # Get train trajectories
            if split_ratio == 1.0:
                dataset_train_trajectories = trajectories
            else:
                split_point = int(len(trajectories) * split_ratio)
                dataset_train_trajectories = trajectories[:split_point]

            print(f"Found {len(dataset_train_trajectories)} train trajectories")

            # Filter and sample train trajectories
            print(f"   Filtering train trajectories by length ({min_length}-{max_length})...")
            filtered_train_trajectories = filter_trajectories_by_length(
                dataset_train_trajectories, min_length, max_length
            )
            print(f"   Filtered train trajectories: {len(filtered_train_trajectories)}")

            if len(filtered_train_trajectories) > num_train_trajectories:
                sampled_train_trajectories = random.sample(filtered_train_trajectories, num_train_trajectories)
                print(f"   Sampled {num_train_trajectories} train trajectories from {len(filtered_train_trajectories)}")
            else:
                sampled_train_trajectories = filtered_train_trajectories
                print(f"   Using all {len(filtered_train_trajectories)} filtered train trajectories")

            if sampled_train_trajectories:
                # Add dataset prefix to trajectory paths
                for traj in sampled_train_trajectories:
                    traj["dataset"] = dataset_name
                    # Store original path temporarily (will be removed before saving to JSON)
                    traj["_original_path"] = traj["path"]
                    traj["path"] = f"{dataset_name}/{traj['path']}"

                train_trajectories_by_dataset[dataset_name] = sampled_train_trajectories
                print(f"   ✅ Added {len(sampled_train_trajectories)} train trajectories from {dataset_name}")
            else:
                print(f"   ❌ No suitable train trajectories found for {dataset_name}")

        # Process test split if applicable
        if split_ratio < 1.0:
            print(f"Processing {dataset_name} test split (ratio: {1.0 - split_ratio})")

            # Get test trajectories
            if split_ratio == 0.0:
                dataset_test_trajectories = trajectories
            else:
                split_point = int(len(trajectories) * split_ratio)
                dataset_test_trajectories = trajectories[split_point:]

            print(f"Found {len(dataset_test_trajectories)} test trajectories")

            # Filter and sample test trajectories
            print(f"   Filtering test trajectories by length ({min_length}-{max_length})...")
            filtered_test_trajectories = filter_trajectories_by_length(
                dataset_test_trajectories, min_length, max_length
            )
            print(f"   Filtered test trajectories: {len(filtered_test_trajectories)}")

            if len(filtered_test_trajectories) > num_test_trajectories:
                sampled_test_trajectories = random.sample(filtered_test_trajectories, num_test_trajectories)
                print(f"   Sampled {num_test_trajectories} test trajectories from {len(filtered_test_trajectories)}")
            else:
                sampled_test_trajectories = filtered_test_trajectories
                print(f"   Using all {len(filtered_test_trajectories)} filtered test trajectories")

            if sampled_test_trajectories:
                # Add dataset prefix to trajectory paths
                for traj in sampled_test_trajectories:
                    traj["dataset"] = dataset_name
                    # Store original path temporarily (will be removed before saving to JSON)
                    traj["_original_path"] = traj["path"]
                    traj["path"] = f"{dataset_name}/{traj['path']}"

                test_trajectories_by_dataset[dataset_name] = sampled_test_trajectories
                print(f"   ✅ Added {len(sampled_test_trajectories)} test trajectories from {dataset_name}")
            else:
                print(f"   ❌ No suitable test trajectories found for {dataset_name}")

    # Combine all train trajectories
    all_train_trajectories = []
    for dataset_name, trajectories in train_trajectories_by_dataset.items():
        all_train_trajectories.extend(trajectories)

    # Combine all test trajectories
    all_test_trajectories = []
    for dataset_name, trajectories in test_trajectories_by_dataset.items():
        all_test_trajectories.extend(trajectories)

    # Process train trajectories
    if all_train_trajectories:
        print(f"\nProcessing all train trajectories ({len(all_train_trajectories)} total)")

        # Duplicate train trajectories
        for traj in tqdm(all_train_trajectories):
            dataset_name = traj["dataset"]
            original_path = traj["_original_path"]
            new_path = traj["path"]

            # Source path
            data_folder = data_configs[dataset_name]["data_folder"]
            src_traj_path = os.path.join(data_folder, original_path)

            # Destination path
            dst_traj_path = os.path.join(train_traj_dir, new_path)

            # Skip if already exists
            if os.path.exists(dst_traj_path):
                continue

            # Create directory
            os.makedirs(dst_traj_path, exist_ok=True)

            # Copy all files and subdirectories
            for item in os.listdir(src_traj_path):
                src_item = os.path.join(src_traj_path, item)
                dst_item = os.path.join(dst_traj_path, item)

                if os.path.isdir(src_item):
                    shutil.copytree(src_item, dst_item)
                else:
                    shutil.copy2(src_item, dst_item)

        # Calculate total image count
        train_img_cnt = sum(traj["cnt"] for traj in all_train_trajectories)

        # Remove temporary original path from trajectories
        for traj in all_train_trajectories:
            if "_original_path" in traj:
                del traj["_original_path"]

        # Create cache metadata for train
        train_metadata = {
            "name": "Viz_data",
            "root": "Viz_data",
            "path": "Viz_data/train_viz",
            "traj_cnt": len(all_train_trajectories),
            "img_cnt": train_img_cnt,
            "trajectories": all_train_trajectories
        }

        # Save cache metadata
        with open(train_metadata_path, "w") as f:
            json.dump(train_metadata, f, indent=2)

        # Build LMDB for train
        build_lmdb(all_train_trajectories, train_traj_dir, train_lmdb_dir)

        print(f"Created unified train visualization cache with {len(all_train_trajectories)} trajectories")

    # Process test trajectories
    if all_test_trajectories:
        print(f"\nProcessing all test trajectories ({len(all_test_trajectories)} total)")

        # Duplicate test trajectories
        for traj in tqdm(all_test_trajectories):
            dataset_name = traj["dataset"]
            original_path = traj["_original_path"]
            new_path = traj["path"]

            # Source path
            data_folder = data_configs[dataset_name]["data_folder"]
            src_traj_path = os.path.join(data_folder, original_path)

            # Destination path
            dst_traj_path = os.path.join(test_traj_dir, new_path)

            # Skip if already exists
            if os.path.exists(dst_traj_path):
                continue

            # Create directory
            os.makedirs(dst_traj_path, exist_ok=True)

            # Copy all files and subdirectories
            for item in os.listdir(src_traj_path):
                src_item = os.path.join(src_traj_path, item)
                dst_item = os.path.join(dst_traj_path, item)

                if os.path.isdir(src_item):
                    shutil.copytree(src_item, dst_item)
                else:
                    shutil.copy2(src_item, dst_item)

        # Calculate total image count
        test_img_cnt = sum(traj["cnt"] for traj in all_test_trajectories)

        # Remove temporary original path from trajectories
        for traj in all_test_trajectories:
            if "_original_path" in traj:
                del traj["_original_path"]

        # Create cache metadata for test
        test_metadata = {
            "name": "Viz_data",
            "root": "Viz_data",
            "path": "Viz_data/test_viz",
            "traj_cnt": len(all_test_trajectories),
            "img_cnt": test_img_cnt,
            "trajectories": all_test_trajectories
        }

        # Save cache metadata
        with open(test_metadata_path, "w") as f:
            json.dump(test_metadata, f, indent=2)

        # Build LMDB for test
        build_lmdb(all_test_trajectories, test_traj_dir, test_lmdb_dir)

        print(f"Created unified test visualization cache with {len(all_test_trajectories)} trajectories")

def filter_trajectories_by_length(trajectories, min_length, max_length):
    """
    Filter trajectories by length.

    Args:
        trajectories: List of trajectories
        min_length: Minimum trajectory length
        max_length: Maximum trajectory length

    Returns:
        List of filtered trajectories
    """
    filtered_trajectories = []
    length_stats = {"too_short": 0, "too_long": 0, "valid": 0}

    for traj in trajectories:
        traj_cnt = traj.get("cnt", 0)

        # Check if trajectory is within length bounds
        if traj_cnt < min_length:
            length_stats["too_short"] += 1
        elif traj_cnt > max_length:
            length_stats["too_long"] += 1
        else:
            filtered_trajectories.append(traj)
            length_stats["valid"] += 1

    print(f"     Length filtering results:")
    print(f"     - Too short (<{min_length}): {length_stats['too_short']}")
    print(f"     - Too long (>{max_length}): {length_stats['too_long']}")
    print(f"     - Valid ({min_length}-{max_length}): {length_stats['valid']}")

    return filtered_trajectories

def has_camera_metrics(config):
    """Check if dataset has camera metrics in config."""
    return ("camera_metrics" in config and
            "camera_matrix" in config["camera_metrics"])


def build_lmdb(samples, traj_dir, lmdb_dir):
    """
    Build LMDB for faster access to images and metadata.

    Args:
        samples: List of sample metadata (in dataset_metadata.json format)
        traj_dir: Directory containing duplicated trajectories
        lmdb_dir: Directory to create LMDB in
    """
    print("Building LMDB cache...")

    # Create LMDB environment
    env = lmdb.open(lmdb_dir, map_size=int(1e12))

    # Add samples to LMDB
    with env.begin(write=True) as txn:
        # Store metadata
        txn.put(b'__metadata__', pickle.dumps({
            'num_samples': len(samples),
            'sample_names': [sample['path'].split('/')[-1] if '/' in sample['path'] else os.path.basename(sample['path']) for sample in samples]
        }))

        # Store each sample
        for i, sample in enumerate(tqdm(samples)):
            traj_path = sample['path']
            # For the unified visualization dataset, the path includes the dataset name
            traj_name = traj_path.split('/')[-1] if '/' in traj_path else os.path.basename(traj_path)
            full_traj_path = os.path.join(traj_dir, traj_path)

            # Load trajectory data
            traj_data_path = os.path.join(full_traj_path, "traj_data.json")
            if not os.path.exists(traj_data_path):
                continue

            with open(traj_data_path, "r") as f:
                traj_data = json.load(f)

            # Store sample metadata
            txn.put(f'sample_{i}_meta'.encode(), pickle.dumps({
                'trajectory_name': traj_name,
                'positions': traj_data.get('position', []),
                'yaws': traj_data.get('yaw', [])
            }))

            # Find all image files in the trajectory directory
            image_files = []
            for root, _, files in os.walk(full_traj_path):
                for file in files:
                    if file.endswith(('.jpg', '.png')):
                        rel_path = os.path.relpath(os.path.join(root, file), full_traj_path)
                        image_files.append(rel_path)

            # Sort image files to ensure consistent ordering
            image_files.sort()

            # Store images
            for j, img_file in enumerate(image_files):
                img_path = os.path.join(full_traj_path, img_file)
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        txn.put(f'sample_{i}_image_{j}'.encode(), pickle.dumps(img))

    env.close()
    print(f"Built LMDB with {len(samples)} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build visualization cache for datasets")
    parser.add_argument("--num-train-trajectories", type=int, default=5, help="Number of trajectories to sample per dataset for train split")
    parser.add_argument("--num-test-trajectories", type=int, default=5, help="Number of trajectories to sample per dataset for test split")
    parser.add_argument("--min-length", type=int, default=10, help="Minimum trajectory length")
    parser.add_argument("--max-length", type=int, default=200, help="Maximum trajectory length")
    args = parser.parse_args()

    build_visualization_cache(
        num_train_trajectories=args.num_train_trajectories,
        num_test_trajectories=args.num_test_trajectories,
        min_length=args.min_length,
        max_length=args.max_length
    )

    print("\nVisualization cache built successfully!")