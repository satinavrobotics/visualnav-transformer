import argparse
import os
import time
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
from visualnav_transformer.train.vint_train.models.nomad.ft_extractor import DiNOV2Extractor
from visualnav_transformer import ROOT_TRAIN

# Load data config
with open(os.path.join(ROOT_TRAIN, "vint_train/data/data_config.yaml"), "r") as f:
    data_configs = yaml.safe_load(f)

def build_dino_feature_cache(
    dataset_dir: str,
    cache_dir: str,
    dino_model_type: str = "large",
    feature_chunk_size: int = 200,
    feature_chunk_overlap: int = 10,
    min_trajectory_length: int = 10,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dataset: ViNT_Dataset = None,
    keep_pt_files: bool = True,
    force_rebuild: bool = False,
):
    """
    Build a cache of DINO features from trajectory images.

    This function processes trajectory images through a DINO v2 model,
    extracts features, and saves them as .pt files in a cache structure.

    Args:
        dataset_dir: Path to the dataset directory
        cache_dir: Path to save the feature cache
        dino_model_type: DINO model type ('small', 'base', 'large', 'giant')
        feature_chunk_size: Number of features in each chunk
        feature_chunk_overlap: Overlap between consecutive chunks
        min_trajectory_length: Minimum number of images in a trajectory
        batch_size: Batch size for feature extraction
        device: Device to use for feature extraction
        dataset: ViNT_Dataset instance (if already created)
        keep_pt_files: Whether to keep the .pt files after creating the LMDB cache
        force_rebuild: Whether to force rebuilding the cache even if it already exists
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Check if cache already exists
    existing_pt_files = glob.glob(os.path.join(cache_dir, "*.pt"))
    existing_lmdb = os.path.exists(os.path.join(cache_dir, "lmdb_cache"))

    # Initialize DINO feature extractor if needed
    feature_extractor = None
    if force_rebuild or not existing_pt_files:
        print(f"Initializing DINO feature extractor with backbone: {dino_model_type}")
        feature_extractor = DiNOV2Extractor(backbone_size=dino_model_type)
        feature_extractor.to(device)
        feature_extractor.eval()

    # Load dataset metadata to include in cache metadata
    dataset_metadata_path = os.path.join(dataset_dir, "dataset_metadata.json")
    if os.path.exists(dataset_metadata_path):
        with open(dataset_metadata_path, "r") as f:
            dataset_metadata = json.load(f)

        # Copy only the basic dataset info fields, not trajectories
        cache_metadata = {
            "name": dataset_metadata.get("name", os.path.basename(dataset_dir)),
            "root": dataset_metadata.get("root", dataset_dir),
            "traj_cnt": dataset_metadata.get("traj_cnt", 0),
            "img_cnt": dataset_metadata.get("img_cnt", 0),
            "duration": dataset_metadata.get("duration", 0)
        }
    else:
        # Initialize cache metadata without dataset info
        cache_metadata = {
            "name": os.path.basename(dataset_dir),
            "root": dataset_dir,
            "traj_cnt": 0,
            "img_cnt": 0,
            "duration": 0
        }

    # Add DINO cache specific fields
    cache_metadata.update({
        "feature_chunk_size": feature_chunk_size,
        "feature_chunk_overlap": feature_chunk_overlap,
        "dino_model_type": dino_model_type,
        "min_trajectory_length": min_trajectory_length,
        "cache_path": os.path.basename(cache_dir),  # Just the folder name, e.g., "dino_cache_large"
        "chunks": []
    })

    # If we have existing .pt files, collect their chunk_ids
    if existing_pt_files and not force_rebuild:
        print(f"Found {len(existing_pt_files)} existing .pt files, collecting chunk IDs...")

        # Use a set to automatically handle duplicates
        chunk_ids_set = set()
        duplicates_found = 0

        for pt_file in tqdm(existing_pt_files, desc="Collecting chunk IDs"):
            # Extract just the filename without extension
            chunk_id = os.path.basename(pt_file)[:-3]  # Remove .pt extension

            if chunk_id in chunk_ids_set:
                duplicates_found += 1
                print(f"Warning: Duplicate chunk ID found: {chunk_id}")
            else:
                chunk_ids_set.add(chunk_id)

        # Convert set back to list
        cache_metadata["chunks"] = list(chunk_ids_set)

        print(f"Collected {len(cache_metadata['chunks'])} unique chunk IDs from existing .pt files")
        if duplicates_found > 0:
            print(f"Warning: Found {duplicates_found} duplicate chunk IDs (removed from metadata)")

    # Process each trajectory only if we need to create new .pt files
    feature_dim = None  # Initialize feature dimension variable
    if not existing_pt_files or force_rebuild:
        for traj_name in tqdm(dataset.traj_names, desc="Processing trajectories"):
            # Get trajectory data
            traj_data = dataset._get_trajectory(traj_name)
            if traj_data is None:
                print(f"Trajectory data not found for {traj_name}")
                continue

            # Get the number of images in the trajectory
            traj_len = len(traj_data["position"])

            # Skip trajectories that are too short
            if traj_len < min_trajectory_length:
                print(f"Trajectory {traj_name} has fewer than {min_trajectory_length} images, skipping.")
                continue

            # Initialize feature extractor if it hasn't been initialized yet
            if feature_extractor is None:
                print(f"Initializing DINO feature extractor with backbone: {dino_model_type}")
                feature_extractor = DiNOV2Extractor(backbone_size=dino_model_type)
                feature_extractor.to(device)
                feature_extractor.eval()

            # Process the trajectory
            chunk_ids, traj_feature_dim = process_trajectory(
                dataset,
                traj_name,
                feature_extractor,
                feature_chunk_size,
                feature_chunk_overlap,
                cache_dir,
                batch_size,
                device
            )

            # Set feature_dim from the first trajectory processed
            if feature_dim is None and traj_feature_dim is not None:
                feature_dim = traj_feature_dim

            # Add chunk IDs to the metadata (prevent duplicates)
            if chunk_ids:
                existing_chunks_set = set(cache_metadata["chunks"])
                new_chunks_added = 0
                for chunk_id in chunk_ids:
                    if chunk_id not in existing_chunks_set:
                        cache_metadata["chunks"].append(chunk_id)
                        existing_chunks_set.add(chunk_id)
                        new_chunks_added += 1
                if new_chunks_added > 0:
                    print(f"Added {new_chunks_added} new chunks for trajectory {traj_name}")

    # Write the feature dimension into the metadata
    if feature_dim is None:                       # warm-start path
        if cache_metadata["chunks"]:
           first_pt = os.path.join(
               cache_dir, cache_metadata["chunks"][0] + ".pt"
            )
           tmp = torch.load(first_pt, map_location="cpu")
           feature_dim = tmp["features"].shape[-1]
           del tmp
        else:
            raise RuntimeError(
                "No chunks found – cannot infer feature dimension."
            )

    cache_metadata["feature_dim"] = feature_dim

    # Save cache metadata
    cache_metadata_path = os.path.join(cache_dir, "cache_metadata.json")
    with open(cache_metadata_path, "w") as f:
        json.dump(cache_metadata, f, indent=2)

    print(f"DINO feature cache built successfully. Metadata saved to {cache_metadata_path}")
    print(f"Total chunks collected: {len(cache_metadata['chunks'])}")

    # Create LMDB environment for faster access (only if it doesn't exist or force rebuild)
    lmdb_path = os.path.join(cache_dir, "lmdb_cache")
    force_lmdb_rebuild = False  # Default to False since config is not available in this scope

    if not os.path.exists(lmdb_path) or force_lmdb_rebuild:
        if force_lmdb_rebuild:
            print("Force rebuilding LMDB cache...")
        else:
            print("LMDB cache not found, creating it...")
        create_lmdb_cache(cache_dir, cache_metadata, keep_pt_files=keep_pt_files)
    else:
        print(f"LMDB cache already exists at {lmdb_path}, skipping creation")

def process_trajectory(
    dataset: ViNT_Dataset,
    trajectory_name: str,
    feature_extractor: DiNOV2Extractor,
    feature_chunk_size: int,
    feature_chunk_overlap: int,
    cache_dir: str,
    batch_size: int,
    device: str
) -> tuple:
    """
    Process a single trajectory, extract features, and create chunks.

    Args:
        dataset: ViNT_Dataset instance
        trajectory_name: Name of the trajectory
        feature_extractor: DINO feature extractor
        feature_chunk_size: Number of features in each chunk
        feature_chunk_overlap: Overlap between consecutive chunks
        cache_dir: Path to save the feature cache
        batch_size: Batch size for feature extraction
        device: Device to use for feature extraction

    Returns:
        Tuple of (chunk_ids, feature_dim) or (None, None) if skipped
    """
    # Get trajectory data
    traj_data = dataset._get_trajectory(trajectory_name)
    traj_len = len(traj_data["position"])

    # Get the trajectory directory
    traj_dir = trajectory_name

    # Load images
    images = []
    valid_positions = []
    valid_indices = []  # Keep track of valid indices

    # Debug: Print first few expected paths to see what's wrong
    if len(images) == 0:  # Only print for the first trajectory
        print(f"Debug: Checking trajectory {trajectory_name}")
        print(f"Debug: Dataset data folder: {dataset.data_folder}")
        print(f"Debug: Trajectory length: {traj_len}")

        # Check first few image paths
        for debug_i in range(min(5, traj_len)):
            from visualnav_transformer.train.vint_train.data.data_utils import get_data_path
            debug_path = get_data_path(dataset.data_folder, trajectory_name, debug_i, "image")
            exists = os.path.exists(debug_path)
            print(f"Debug: Image {debug_i}: {debug_path} (exists: {exists})")

            # If path doesn't exist, try to find what files actually exist
            if not exists:
                if os.path.exists(traj_dir):
                    files = os.listdir(traj_dir)[:10]  # First 10 files
                    print(f"Debug: Files in trajectory dir: {files}")
                else:
                    print(f"Debug: Trajectory directory doesn't exist: {traj_dir}")
                break

    # only keep files that are not JSON or PKL
    files_sorted = sorted(
        f
        for f in os.listdir(traj_dir)
        if not (f.endswith(".json") or f.endswith(".pkl"))
    )
    assert len(files_sorted) == traj_len, f"Expected {traj_len} images, found {len(files_sorted)}"
    for i, file_name in enumerate(files_sorted):
        try:
            # Load image directly from filesystem instead of using dataset's LMDB cache
            from visualnav_transformer.train.vint_train.data.data_utils import get_data_path, img_path_to_data

            # Get the image path directly
            image_path = os.path.join(traj_dir, file_name) # get_data_path(dataset.data_folder, trajectory_name, i, "image")

            # Check if the image file exists
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                continue

            # Load image directly from file
            img_tensor = img_path_to_data(image_path, dataset.image_size)
            if img_tensor is None:
                continue

            images.append(img_tensor)
            valid_indices.append(i)  # Store the valid index

            # Extract position data
            if i < len(traj_data["position"]):
                pos = traj_data["position"][i]
                yaw = traj_data["yaw"][i] if "yaw" in traj_data and i < len(traj_data["yaw"]) else 0
                valid_positions.append({"position": pos, "yaw": yaw})
            else:
                # If position data is missing, use the last known position
                valid_positions.append(valid_positions[-1] if valid_positions else {"position": [0, 0, 0], "yaw": 0})
        except Exception as e:
            print(f"Error processing image at index {i} for trajectory {trajectory_name}: {e}")
            continue

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
    if not all_features:
        print(f"No valid features extracted for trajectory {trajectory_name}, skipping.")
        return None, None

    features = torch.cat(all_features, dim=0)

    # Create feature chunks
    chunks = create_feature_chunks(
        features,
        valid_positions,
        valid_indices,
        trajectory_name,
        traj_data,  # Pass the full trajectory data
        feature_chunk_size,
        feature_chunk_overlap,
        cache_dir
    )

    # Return the list of chunk IDs and feature dimension
    return chunks, features.shape[-1]

def create_feature_chunks(
    features: torch.Tensor,
    positions: List[Dict],
    valid_indices: List[int],
    trajectory_id: str,
    full_traj_data: Dict,
    feature_chunk_size: int,
    feature_chunk_overlap: int,
    cache_dir: str
) -> List[Dict]:
    """
    Create feature chunks with specified size and overlap.

    Args:
        features: Tensor of features for a trajectory
        positions: List of position data for each image
        valid_indices: List of valid indices in the original trajectory
        trajectory_id: ID of the trajectory
        full_traj_data: Full trajectory data from traj_data.json
        feature_chunk_size: Number of features in each chunk
        feature_chunk_overlap: Overlap between consecutive chunks
        cache_dir: Path to save the feature cache

    Returns:
        List of dictionaries with chunk metadata
    """
    chunks = []
    stride = feature_chunk_size - feature_chunk_overlap

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Calculate the starting indices for chunks
    # We'll manually handle the last chunk to avoid overlapping
    start_indices = []
    current_idx = 0

    while current_idx + feature_chunk_size <= len(features):
        start_indices.append(current_idx)
        current_idx += stride

    # Add the last chunk if there are remaining features
    if current_idx < len(features) and len(features) - current_idx > feature_chunk_overlap:
        start_indices.append(current_idx)

    # Process each chunk
    for start_idx in start_indices:
        # Calculate end index, but don't go beyond the end of the features
        end_idx = min(start_idx + feature_chunk_size, len(features))

        # Extract features and positions for this chunk
        chunk_features = features[start_idx:end_idx]
        chunk_positions = positions[start_idx:end_idx]

        # Extract the corresponding indices in the original trajectory
        chunk_indices = valid_indices[start_idx:end_idx]

        # Extract the image directory name (the part after the base path)
        # For example, from "A_Jackal_GDC_GDC_Fri_Oct_29_11/rgb_320x240_camera_rgb_image_raw_compressed"
        # We want to extract: "rgb_320x240_camera_rgb_image_raw_compressed"
        img_dir = trajectory_id.split('/')[-1] if '/' in trajectory_id else trajectory_id

        # Create a unique chunk ID using just the image directory and index
        chunk_id = f"{img_dir}_{start_idx:05d}"

        # Create the full path for the .pt file
        chunk_path = os.path.join(cache_dir, f"{chunk_id}.pt")

        # Create the new slim chunk layout
        chunk_traj_data = {
            "features": chunk_features.half().contiguous(),   # (L, D) fp16
            "positions": chunk_positions,                     # [{'position':…, 'yaw':…}, …]
            "indices":   chunk_indices                        # [global_frame_numbers]
        }

        # Save the features, positions, and trajectory data
        # Note: We're saving dictionaries with tensor values, which requires weights_only=False when loading
        torch.save(chunk_traj_data, chunk_path)

        print(f"Saved feature chunk to: {chunk_path} (size: {len(chunk_features)})")

        # Add just the chunk_id to the list
        chunks.append(chunk_id)

    return chunks

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

def main(config):
    """
    Main function to build a DINO feature cache based on the provided configuration.

    Args:
        config: Configuration dictionary
    """
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])

    # Create datasets
    datasets = []

    if "context_type" not in config:
        config["context_type"] = "temporal"

    if "clip_goals" not in config:
        config["clip_goals"] = False

    # ——— 1) Normal datasets (everything except viz_data) ———
    for dataset_index, dataset_name in enumerate(data_configs["datasets"]):
        if dataset_name == "viz_data":
            continue
        data_config = data_configs["datasets"][dataset_name]
        if not data_config.get("available", False):
            continue

        print(f"\n📦 Building DINO cache for dataset: {dataset_name}")
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
            metric_waypoint_spacing= data_config["metric_waypoint_spacing"],
            min_dist_cat           = config["distance"]["min_dist_cat"],
            max_dist_cat           = config["distance"]["max_dist_cat"],
            min_action_distance    = config["action"]["min_dist_cat"],
            max_action_distance    = config["action"]["max_dist_cat"],
            negative_mining        = data_config["negative_mining"],
            len_traj_pred          = config["len_traj_pred"],
            learn_angle            = config["learn_angle"],
            context_size           = config["context_size"],
            end_slack              = data_config["end_slack"],
            goals_per_obs          = data_config["goals_per_obs"],
            normalize               = config["normalize"],
            build_image_cache      = False,
        )

        cache_dir = os.path.join(
            data_config["data_folder"],
            f"dino_cache_{config['dino_model_type']}"
        )
        build_dino_feature_cache(
            dataset_dir           = data_config["data_folder"],
            cache_dir             = cache_dir,
            dino_model_type       = config["dino_model_type"],
            feature_chunk_size    = config["feature_chunk_size"],
            feature_chunk_overlap = config["feature_chunk_overlap"],
            min_trajectory_length = config["min_trajectory_length"],
            batch_size            = config["batch_size"],
            device                = device,
            dataset               = ds,
            keep_pt_files         = config["keep_pt_files"],
        )

    # ——— 2) Now build for Viz_data/train_viz & Viz_data/test_viz ———
    viz_cfg = data_configs["datasets"].get("viz_data", {})
    if viz_cfg.get("available", False):
        viz_root = viz_cfg["data_folder"]
        for split in ("train_viz","test_viz"):
            folder = os.path.join(viz_root, split)
            if not os.path.isdir(folder):
                print(f"[warning] missing viz split: {folder}")
                continue
            print(f"\n🔨 Building DINO cache for Viz_data/{split}")
            viz_ds = ViNT_Dataset(
                data_folder     = folder,
                split           = "train",
                split_ratio     = 1.0,
                dataset_name    = f"viz_{split}",
                dataset_index   = 0,  # doesn’t matter
                image_size      = config["image_size"],
                waypoint_spacing       = viz_cfg.get("waypoint_spacing",1),
                metric_waypoint_spacing= viz_cfg.get("metric_waypoint_spacing",1.0),
                min_dist_cat           = config["distance"]["min_dist_cat"],
                max_dist_cat           = config["distance"]["max_dist_cat"],
                min_action_distance    = config["action"]["min_dist_cat"],
                max_action_distance    = config["action"]["max_dist_cat"],
                negative_mining        = viz_cfg.get("negative_mining",True),
                len_traj_pred          = config["len_traj_pred"],
                learn_angle            = config["learn_angle"],
                context_size           = config["context_size"],
                end_slack              = viz_cfg.get("end_slack",0),
                goals_per_obs          = viz_cfg.get("goals_per_obs",1),
                normalize               = config["normalize"],
                build_image_cache      = False,
            )
            cache_dir = os.path.join(folder, f"dino_cache_{config['dino_model_type']}")
            build_dino_feature_cache(
                dataset_dir           = folder,
                cache_dir             = cache_dir,
                dino_model_type       = config["dino_model_type"],
                feature_chunk_size    = config["feature_chunk_size"],
                feature_chunk_overlap = config["feature_chunk_overlap"],
                min_trajectory_length = config["min_trajectory_length"],
                batch_size            = config["batch_size"],
                device                = device,
                dataset               = viz_ds,
                keep_pt_files         = config["keep_pt_files"],
            )

    print("✅ DINO feature cache built for all datasets & viz_data!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINO Feature Cache Builder")

    # project setup
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

    # Add DINO cache specific defaults if not present
    if "dino_model_type" not in config:
        config["dino_model_type"] = "large"
    if "feature_chunk_size" not in config:
        config["feature_chunk_size"] = 200
    if "feature_chunk_overlap" not in config:
        config["feature_chunk_overlap"] = 10
    if "min_trajectory_length" not in config:
        config["min_trajectory_length"] = 10
    if "batch_size" not in config:
        config["batch_size"] = 32
    if "keep_pt_files" not in config:
        config["keep_pt_files"] = True  # Always keep .pt files by default

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    main(config)