import argparse
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import yaml
import lmdb
import pickle
import shutil
import glob
from typing import Dict, List, Optional
# No need for these imports

# Import existing classes and utilities
from visualnav_transformer.train.vint_train.data.vint_dataset import ViNT_Dataset
from visualnav_transformer.train.vint_train.models.ft_extractor import DiNOV2Extractor
from visualnav_transformer import ROOT_TRAIN
from visualnav_transformer.train.vint_train.data.data_utils import img_path_to_data

from dataset.preprocess.stage_4.dino_cache_utils import (
    calculate_distance_meters,
    calculate_trajectory_advancement,
    setup_gpu,
    create_meter_based_chunks,
    calculate_fps_scaling_factor,
)

# Load data config
with open(os.path.join(ROOT_TRAIN, "vint_train/data/data_config.yaml"), "r") as f:
    data_configs = yaml.safe_load(f)


def process_trajectory(
    dataset: ViNT_Dataset,
    trajectory_name: str,
    feature_extractor: DiNOV2Extractor,
    max_chunk_distance_m: float,
    overlap_distance_m: float,
    min_chunk_distance_m: float,
    cache_dir: str,
    batch_size: int,
    device: str,
    fps_estimates: Optional[Dict] = None,
    dataset_name: str = ""
) -> tuple:
    """
    Process a single trajectory, extract features, and create meter-based chunks.

    Args:
        dataset: ViNT_Dataset instance
        trajectory_name: Name of the trajectory
        feature_extractor: DINO feature extractor
        max_chunk_distance_m: Maximum distance per chunk in meters
        overlap_distance_m: Overlap distance between chunks in meters
        min_chunk_distance_m: Minimum distance for a chunk in meters
        cache_dir: Path to save the feature cache
        batch_size: Batch size for feature extraction
        device: Device to use for feature extraction
        fps_estimates: FPS estimates from json file
        dataset_name: Name of dataset for FPS scaling

    Returns:
        Tuple of (chunk_ids, feature_dim) or (None, None) if skipped
    """
    # Get trajectory data BEFORE scaling
    traj_data = dataset._get_trajectory(trajectory_name)
    assert "position" in traj_data and "yaw" in traj_data
    traj_len = len(traj_data["position"])
    images_sorted = sorted(
        f for f in os.listdir(trajectory_name)
        if not (f.endswith(".json") or f.endswith(".pkl"))
    )
    indices = list(range(len(images_sorted)))
    positions = traj_data["position"]
    yaws = traj_data["yaw"]
    

    # Apply FPS scaling FIRST, before trajectory advancement check
    fps_scale_factor = calculate_fps_scaling_factor(dataset_name, fps_estimates)
    if fps_scale_factor > 1:
        original_traj_len = traj_len
        print(f"Applying FPS scaling for {dataset_name} dataset: keeping every {fps_scale_factor}th frame")
        # Scale trajectory data
        positions = positions[::fps_scale_factor]
        yaws = yaws[::fps_scale_factor]
        traj_data = {"position": positions, "yaw": yaws}
        traj_len = len(positions)
        images_sorted = images_sorted[::fps_scale_factor]
        indices = list(range(0, original_traj_len, fps_scale_factor))[:len(images_sorted)]
        print(f"{dataset_name} trajectory scaled from {original_traj_len} to {traj_len} frames")
    

    assert len(images_sorted) == traj_len, \
            f"Warning: Image count ({len(images_sorted)}) doesn't match trajectory length ({traj_len}) after scaling" + \
            f"Make the proper manual corrections in {dataset_name}/{trajectory_name}"

    # Load images with consistent scaling
    images = []
    for i, file_name in enumerate(images_sorted):
        image_path = os.path.join(trajectory_name, file_name)
        assert os.path.exists(image_path), f"Image file not found: {image_path}"
        # Load image directly from file
        img_tensor = img_path_to_data(image_path, dataset.image_size)
        assert img_tensor is not None, f"Failed to load image: {image_path}"
        images.append(img_tensor)

    # Extract features in batches
    all_features = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        with torch.no_grad():
            batch = torch.stack(batch_images).to(device)
            # Use the extract_features method from DiNOV2Extractor
            batch_features = feature_extractor.extract_features(batch)
            all_features.append(batch_features.cpu())

    # Concatenate all features
    assert len(all_features) > 0, f"No valid features extracted for trajectory {trajectory_name}, skipping."
    features = torch.cat(all_features, dim=0)

    # Create feature chunks using meter-based slicing
    chunks = create_feature_chunks_meter_based(
        features,
        traj_data,
        indices,
        trajectory_name,
        cache_dir,
        max_chunk_distance_m,
        overlap_distance_m,
        min_chunk_distance_m
    )

    # Return the list of chunk IDs and feature dimension
    return chunks, features.shape[-1]


def create_feature_chunks_meter_based(
    features: torch.Tensor,
    traj_data: List[Dict],
    valid_indices: List[int],
    trajectory_id: str,
    cache_dir: str,
    max_chunk_distance_m: float = 10.0,
    overlap_distance_m: float = 1.0,
    min_chunk_distance_m: float = 0.3
) -> List[str]:
    """
    Create feature chunks based on meter-based advancement instead of frame count.

    Args:
        features: Tensor of features for a trajectory
        positions: List of position data for each image
        valid_indices: List of valid indices in the original trajectory
        trajectory_id: ID of the trajectory
        full_traj_data: Full trajectory data from traj_data.json
        cache_dir: Path to save the feature cache
        max_chunk_distance_m: Maximum distance per chunk (default 10m)
        overlap_distance_m: Overlap distance between chunks (default 1m)
        min_chunk_distance_m: Minimum distance for a chunk (default 0.3m)

    Returns:
        List of chunk IDs
    """

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    positions = traj_data["position"]
    position_coords = [pos[:2] for pos in positions]  # Only x, y coordinates

    # Create meter-based chunks
    chunk_ranges = create_meter_based_chunks(
        position_coords,
        max_chunk_distance_m,
        overlap_distance_m,
        min_chunk_distance_m
    )

    # Process each chunk
    CHUNKS = []
    for start_idx, end_idx in chunk_ranges:
        # Extract features and positions for this chunk
        chunk_features = features[start_idx:end_idx+1]  # Include end_idx
        chunk_positions = positions[start_idx:end_idx+1]
        chunk_indices = valid_indices[start_idx:end_idx+1]

        # Extract the image directory name (the part after the base path)
        # For example, from "A_Jackal_GDC_GDC_Fri_Oct_29_11/rgb_320x240_camera_rgb_image_raw_compressed"
        # We want to extract: "rgb_320x240_camera_rgb_image_raw_compressed"
        img_dir = trajectory_id.split('/')[-1] if '/' in trajectory_id else trajectory_id

        # Create a unique chunk ID using just the image directory and starting index
        chunk_id = f"{img_dir}_{start_idx:05d}"

        # Create the full path for the .pt file
        chunk_path = os.path.join(cache_dir, f"{chunk_id}.pt")

        # Create the new slim chunk layout
        # TODO: should we use float32?
        chunk_traj_data = {
            "features": chunk_features.half().contiguous(),   # (L, D) fp16
            "positions": chunk_positions,                     # [{'position':…, 'yaw':…}, …]
            "indices":   chunk_indices                        # [global_frame_numbers]
        }

        # Save the features, positions, and trajectory data
        # Note: We're saving dictionaries with tensor values, which requires weights_only=False when loading
        torch.save(chunk_traj_data, chunk_path)

        chunk_distance = calculate_distance_meters(
            chunk_positions[0][:2],
            chunk_positions[-1][:2]
        )
        # TODO: should we print here?
        #print(f"Saved meter-based chunk to: {chunk_path} (size: {len(chunk_features)}, distance: {chunk_distance:.2f}m)")

        # Add just the chunk_id to the list
        CHUNKS.append(chunk_id)

    return CHUNKS

def create_lmdb_cache(cache_dir: str, cache_metadata: Dict, keep_pt_files: bool = True) -> None:
    """
    Create an LMDB cache from the feature chunks for faster access.

    Args:
        cache_dir: Path to the cache directory
        cache_metadata: Cache metadata dictionary
        keep_pt_files: Whether to keep the .pt files after creating the LMDB cache
    """
    print("Creating LMDB cache for faster access...")

    # Create LMDB environment
    lmdb_path = os.path.join(cache_dir, "lmdb_cache")
    if os.path.exists(lmdb_path):
        shutil.rmtree(lmdb_path)
    os.makedirs(lmdb_path, exist_ok=True)

    # Calculate map size (estimate 50MB per chunk, with no upper limit)
    total_chunks = len(cache_metadata["chunks"])
    # Use a much larger map_size to ensure we can accommodate all chunks
    # 50MB per chunk should be more than enough for DINO features + trajectory data
    map_size = max(total_chunks * 50 * 1024 * 1024, 10 * 1024 * 1024 * 1024)  # At least 10GB

    # Create LMDB environment
    env = lmdb.open(lmdb_path, map_size=map_size)

    # Add chunks to LMDB in batches to prevent memory exhaustion
    LMDB_BATCH_SIZE = 100  # Process 100 chunks at a time
    total_chunks = len(cache_metadata["chunks"])
    chunks_added = 0

    print(f"Processing {total_chunks} chunks in batches of {LMDB_BATCH_SIZE}")

    for batch_start in tqdm(range(0, total_chunks, LMDB_BATCH_SIZE), desc="Processing LMDB batches"):
        batch_end = min(batch_start + LMDB_BATCH_SIZE, total_chunks)
        batch_chunks = cache_metadata["chunks"][batch_start:batch_end]

        # Process this batch in a single transaction
        with env.begin(write=True) as txn:
            for chunk_id in batch_chunks:
                # Reconstruct the path from chunk_id
                chunk_path = os.path.join(cache_dir, f"{chunk_id}.pt")

                # Check if the file exists
                if not os.path.exists(chunk_path):
                    print(f"Warning: Chunk file not found: {chunk_path}")
                    continue

                # Load chunk data with weights_only=False for backward compatibility
                chunk_data = torch.load(chunk_path, weights_only=False, map_location='cpu')

                # Serialize chunk data
                serialized_data = pickle.dumps(chunk_data)

                # Add to LMDB
                txn.put(chunk_id.encode(), serialized_data)

                # Explicit memory cleanup
                del chunk_data, serialized_data
                chunks_added += 1

        # Clear any remaining memory after each batch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Progress update
        print(f"Completed batch {batch_start//LMDB_BATCH_SIZE + 1}/{(total_chunks + LMDB_BATCH_SIZE - 1)//LMDB_BATCH_SIZE}, "
              f"added {chunks_added}/{total_chunks} chunks")

    # Close LMDB environment
    env.close()

    print(f"LMDB cache created at {lmdb_path}")
    print(f"Total chunks added: {chunks_added}")

    # Clean up .pt files if not keeping them
    if not keep_pt_files:
        print("Removing individual .pt files to save space...")
        for chunk_id in tqdm(cache_metadata["chunks"], desc="Cleaning up .pt files"):
            chunk_path = os.path.join(cache_dir, f"{chunk_id}.pt")
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
        print("All .pt files removed. Only LMDB cache remains.")


def build_dino_feature_cache(
    dataset_dir: str,
    cache_dir: str,
    feature_extractor= None,
    max_chunk_distance_m: float = 10.0,
    overlap_distance_m: float = 1.0,
    min_chunk_distance_m: float = 0.3,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dataset: ViNT_Dataset = None,
    keep_pt_files: bool = True,
    force_rebuild: bool = True,
    fps_estimates_path: str = None,
    dataset_name: str = "",
):
    """
    Build a cache of DINO features from trajectory images using meter-based chunking.

    Args:
        dataset_dir: Path to the dataset directory
        cache_dir: Path to save the feature cache
        dino_model_type: DINO model type ('small', 'base', 'large', 'giant')
        max_chunk_distance_m: Maximum distance per chunk in meters (10m)
        overlap_distance_m: Overlap distance between chunks in meters (1m)
        min_chunk_distance_m: Minimum distance for a chunk in meters (0.3m)
        batch_size: Batch size for feature extraction
        device: Device to use for feature extraction
        dataset: ViNT_Dataset instance (if already created)
        keep_pt_files: Whether to keep the .pt files after creating the LMDB cache
        force_rebuild: Whether to force rebuilding the cache even if it already exists
        fps_estimates_path: Path to fps_estimates.json file
        dataset_name: Name of dataset for FPS scaling
    """
    # Load FPS estimates
    fps_estimates = None
    if fps_estimates_path and os.path.exists(fps_estimates_path):
        with open(fps_estimates_path, 'r') as f:
            fps_estimates = json.load(f)
        print(f"Loaded FPS estimates for {len(fps_estimates)} datasets")
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Check if cache already exists
    existing_pt_files = glob.glob(os.path.join(cache_dir, "*.pt"))
    existing_lmdb = os.path.exists(os.path.join(cache_dir, "lmdb_cache"))

    CHUNKS = set()
    # If we have existing .pt files, collect their chunk_ids
    if existing_pt_files and not force_rebuild:
        print(f"Found {len(existing_pt_files)} existing .pt files, collecting chunk IDs...")
        duplicates_found = 0
        for pt_file in tqdm(existing_pt_files, desc="Collecting chunk IDs"):
            # Extract just the filename without extension
            chunk_id = os.path.basename(pt_file)[:-3]  # Remove .pt extension
            if chunk_id in CHUNKS:
                duplicates_found += 1
                print(f"Warning: Duplicate chunk ID found: {chunk_id}")
                continue
            CHUNKS.add(chunk_id)
        print(f"Collected {len(CHUNKS)} chunks from existing .pt files, found {duplicates_found} duplicates")

    # Process each trajectory only if we need to create new .pt files
    feature_dim = None  
    if not existing_pt_files or force_rebuild:
        
        for traj_name in tqdm(dataset.traj_names, desc="Processing trajectories"):
            # Get trajectory data
            traj_data = dataset._get_trajectory(traj_name)
            if traj_data is None:
                print(f"Trajectory data not found for {traj_name}")
                continue

            # Check trajectory advancement in meters instead of frame count
            trajectory_advancement = calculate_trajectory_advancement(traj_data["position"])
            if trajectory_advancement < min_chunk_distance_m:
                print(f"Trajectory {traj_name} has advancement of {trajectory_advancement:.2f}m (< {min_chunk_distance_m}m), skipping.")
                continue

            # Process the trajectory with meter-based parameters and FPS scaling
            chunk_ids, traj_feature_dim = process_trajectory(
                dataset,
                traj_name,
                feature_extractor,
                max_chunk_distance_m,
                overlap_distance_m,
                min_chunk_distance_m,
                cache_dir,
                batch_size,
                device,
                fps_estimates,
                dataset_name
            )

            # Set feature_dim from the first trajectory processed
            if feature_dim is None and traj_feature_dim is not None:
                feature_dim = traj_feature_dim

            # Add chunk IDs to the metadata (prevent duplicates)
            if chunk_ids:
                new_chunks_added = 0
                for chunk_id in chunk_ids:
                    if chunk_id not in CHUNKS:
                        CHUNKS.add(chunk_id)
                        new_chunks_added += 1
                # TODO: should we print here?
                #print(f"Added {new_chunks_added} new chunks for trajectory {traj_name}")

    # Write the feature dimension into the metadata
    if feature_dim is None:                       # warm-start path
        feature_dim = torch.load(os.path.join(
            cache_dir, list(CHUNKS)[0] + ".pt"), map_location="cpu"
        )["features"].shape[-1]

    # Load dataset metadata to include in cache metadata
    dataset_metadata_path = os.path.join(dataset_dir, "dataset_metadata.json")
    with open(dataset_metadata_path, "r") as f:
        dataset_metadata = json.load(f)
    cache_metadata = {
        "name": dataset_metadata.get("name", os.path.basename(dataset_dir)),
        "max_chunk_distance_m": max_chunk_distance_m,
        "overlap_distance_m": overlap_distance_m,
        "min_chunk_distance_m": min_chunk_distance_m,
        "feature_dim": feature_dim,
        "cache_path": os.path.basename(cache_dir),  # Just the folder name, e.g., "dino_cache_large"
        "chunks": list(CHUNKS)
    }
     # Save cache metadata
    cache_metadata_path = os.path.join(cache_dir, "cache_metadata.json")
    with open(cache_metadata_path, "w") as f:
        json.dump(cache_metadata, f, indent=2)

    print(f"DINO feature cache built successfully. Metadata saved to {cache_metadata_path}")
    print(f"Total chunks collected: {len(CHUNKS)}")

    # Create LMDB environment for faster access (only if it doesn't exist or force rebuild)
    #lmdb_path = os.path.join(cache_dir, "lmdb_cache")
    #if not os.path.exists(lmdb_path) or force_rebuild:
    #    print("Building LMDB cache...")
    #    create_lmdb_cache(cache_dir, cache_metadata, keep_pt_files=keep_pt_files)
    #else:
    #    print(f"LMDB cache already exists at {lmdb_path}, skipping creation")



def main(config):
    """
    Main function to build a DINO feature cache based on the provided configuration.

    Args:
        config: Configuration dictionary
    """
    device = setup_gpu(config)
    
    # TODO: make this conditional on force-rebuild
    print(f"Initializing DINO feature extractor with backbone: {config['dino_model_type']}")
    feature_extractor = DiNOV2Extractor(backbone_size=config['dino_model_type'])
    feature_extractor.to(device).eval()

    # ——— 1) Normal datasets (everything except viz_data) ———
    for dataset_index, dataset_name in enumerate(data_configs["datasets"]):
        data_config = data_configs["datasets"][dataset_name]
        if not data_config.get("available", False):
            continue
        
        print(f"\n📦 Building DINO cache for dataset: {dataset_name}")
        
        if dataset_name == "viz_data":
            viz_cfg = data_config
            viz_root = viz_cfg["data_folder"]
            for split in ("train_viz","test_viz"):
                folder = os.path.join(viz_root, split)
                cache_dir = os.path.join(folder, f"dino_cache_{config['dino_model_type']}")
                if not os.path.isdir(folder):
                    print(f"[warning] missing viz split: {folder}")
                    continue
                viz_ds = ViNT_Dataset(
                    data_folder     = folder,
                    split           = "train",
                    split_ratio     = 1.0,
                    dataset_name    = f"viz_{split}",
                    dataset_index   = 0,  # doesn't matter
                    image_size      = config["image_size"],
                    waypoint_spacing       = viz_cfg.get("waypoint_spacing",1),
                    min_goal_distance_meteres=config.get("min_goal_distance_meters",1.0),
                    max_goal_distance_meters=config.get("max_goal_distance_meters",10.0),
                    negative_mining        = viz_cfg.get("negative_mining",True),
                    len_traj_pred          = config["len_traj_pred"],
                    context_size           = config["context_size"],
                    end_slack              = viz_cfg.get("end_slack",0),
                    goals_per_obs          = viz_cfg.get("goals_per_obs",1),
                    normalize               = config["normalize"],
                )
                build_dino_feature_cache(
                    dataset_dir                = folder,
                    cache_dir                  = cache_dir,
                    feature_extractor          = feature_extractor,
                    max_chunk_distance_m       = config["max_chunk_distance_m"],
                    overlap_distance_m         = config["overlap_distance_m"],
                    min_chunk_distance_m       = config["min_chunk_distance_m"],
                    batch_size                 = config["batch_size"],
                    device                     = device,
                    dataset                    = viz_ds,
                    keep_pt_files              = config["keep_pt_files"],
                    fps_estimates_path         = "/home/ubuntu/SatiNav/training_server/Sati_data/fps_estimates.json",
                    dataset_name               = f"viz_{split}",
                )
        else: 
            # fill in any missing defaults
            data_config.setdefault("negative_mining", True)
            data_config.setdefault("goals_per_obs", 1)
            data_config.setdefault("end_slack", 0)
            data_config.setdefault("waypoint_spacing", 1)
            data_config.setdefault("metric_waypoint_spacing", 1.0)

            ds = ViNT_Dataset(
                data_folder            = data_config["data_folder"],
                split                  = "train",
                split_ratio            = 1.0,
                dataset_name           = dataset_name,
                dataset_index          = dataset_index,
                image_size             = config["image_size"],
                waypoint_spacing       = data_config["waypoint_spacing"],
                min_goal_distance_meteres=config["min_goal_distance_meters"],
                max_goal_distance_meters=config["max_goal_distance_meters"],
                negative_mining        = data_config["negative_mining"],
                len_traj_pred          = config["len_traj_pred"],
                context_size           = config["context_size"],
                end_slack              = data_config["end_slack"],
                goals_per_obs          = data_config["goals_per_obs"],
                normalize               = config["normalize"],
            )

            cache_dir = os.path.join(
                data_config["data_folder"],
                f"dino_cache_{config['dino_model_type']}"
            )
            build_dino_feature_cache(
                dataset_dir                = data_config["data_folder"],
                cache_dir                  = cache_dir,
                feature_extractor          = feature_extractor,
                max_chunk_distance_m       = config["max_chunk_distance_m"],
                overlap_distance_m         = config["overlap_distance_m"],
                min_chunk_distance_m       = config["min_chunk_distance_m"],
                batch_size                 = config["batch_size"],
                device                     = device,
                dataset                    = ds,
                keep_pt_files              = config["keep_pt_files"],
                fps_estimates_path         = "/home/ubuntu/SatiNav/training_server/Sati_data/fps_estimates.json",
                dataset_name               = dataset_name,
            )

    print("✅ DINO feature cache built for all datasets & viz_data!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINO Feature Cache Builder")
    parser.add_argument(
        "--config",
        "-c",
        default="config/dino_cache.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    # Load default config
    with open(os.path.join(ROOT_TRAIN, "config/defaults.yaml"), "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    # Load user config
    with open(os.path.join(ROOT_TRAIN, args.config), "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)

    # Add meter-based DINO cache specific defaults if not present
    if "dino_model_type" not in config:
        config["dino_model_type"] = "large"
    if "max_chunk_distance_m" not in config:
        config["max_chunk_distance_m"] = 10.0
    if "overlap_distance_m" not in config:
        config["overlap_distance_m"] = 1.0
    if "min_chunk_distance_m" not in config:
        config["min_chunk_distance_m"] = 0.3
    if "batch_size" not in config:
        config["batch_size"] = 32
    if "keep_pt_files" not in config:
        config["keep_pt_files"] = True

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    main(config)