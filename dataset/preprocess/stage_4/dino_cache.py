import argparse
import os
import json
import torch
from tqdm import tqdm
import yaml
import lmdb
import pickle
import shutil
import glob
import time
import hashlib
from typing import Dict, List, Optional

from visualnav_transformer.train.vint_train.models.ft_extractor import DiNOV2Extractor

# Import existing classes and utilities
from dataset.vint_dataset import ViNT_Dataset
from dataset.data_utils import img_path_to_data
from dataset.preprocess.stage_4.dino_cache_utils import (
    calculate_distance_meters,
    calculate_trajectory_advancement,
    setup_gpu,
    create_meter_based_chunks,
    calculate_fps_scaling_factor,
)

ROOT_DIR = "/app/visualnav-transformer"


class ProgressTracker:
    """
    Robust progress tracking for DINO cache preprocessing.
    Tracks trajectory-level progress and allows resuming from interruptions.
    """

    def __init__(self, cache_dir: str, dataset_name: str = ""):
        self.cache_dir = cache_dir
        self.dataset_name = dataset_name
        self.progress_file = os.path.join(cache_dir, "progress.json")
        self.progress_data = self._load_progress()

    def _load_progress(self) -> Dict:
        """Load existing progress data or create new structure."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                print(f"Loaded existing progress: {len(data.get('completed_trajectories', []))} trajectories completed")
                return data
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load progress file ({e}), starting fresh")

        return {
            "dataset_name": self.dataset_name,
            "start_time": time.time(),
            "last_update": time.time(),
            "completed_trajectories": [],
            "failed_trajectories": [],
            "insufficient_frame_trajectories": {},  # Cache for trajectories with insufficient frames
            "total_chunks": 0,
            "processing_config": {},
            "trajectory_checksums": {}
        }

    def _save_progress(self):
        """Save current progress to disk."""
        self.progress_data["last_update"] = time.time()
        os.makedirs(self.cache_dir, exist_ok=True)

        # Write to temporary file first, then rename for atomic operation
        temp_file = self.progress_file + ".tmp"
        try:
            with open(temp_file, 'w') as f:
                json.dump(self.progress_data, f, indent=2)
            os.rename(temp_file, self.progress_file)
        except Exception as e:
            print(f"Warning: Could not save progress ({e})")
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _calculate_trajectory_checksum(self, traj_data: Dict) -> str:
        """Calculate checksum for trajectory data to detect changes."""
        # Create a simple checksum based on trajectory length and first/last positions
        if not traj_data or "position" not in traj_data:
            return ""

        positions = traj_data["position"]
        if len(positions) == 0:
            return ""

        # Use length, first position, last position, and middle position for checksum
        checksum_data = {
            "length": len(positions),
            "first_pos": positions[0][:2] if len(positions[0]) >= 2 else positions[0],
            "last_pos": positions[-1][:2] if len(positions[-1]) >= 2 else positions[-1]
        }
        if len(positions) > 2:
            mid_idx = len(positions) // 2
            checksum_data["mid_pos"] = positions[mid_idx][:2] if len(positions[mid_idx]) >= 2 else positions[mid_idx]

        checksum_str = json.dumps(checksum_data, sort_keys=True)
        return hashlib.md5(checksum_str.encode()).hexdigest()[:16]

    def is_trajectory_completed(self, traj_name: str, traj_data: Dict = None) -> bool:
        """Check if trajectory has been completed and data hasn't changed (optimized)."""
        # Fast path: if not in completed list, return immediately (no computation needed)
        if traj_name not in self.progress_data["completed_trajectories"]:
            return False

        # Only do expensive checksum validation if trajectory data is provided AND trajectory is marked complete
        if traj_data is not None:
            stored_checksum = self.progress_data["trajectory_checksums"].get(traj_name, "")
            if stored_checksum:  # Only compute if we have a stored checksum to compare
                current_checksum = self._calculate_trajectory_checksum(traj_data)
                if current_checksum != stored_checksum:
                    print(f"Trajectory {traj_name} data has changed, will reprocess")
                    self.mark_trajectory_incomplete(traj_name)
                    return False

        return True

    def should_skip_trajectory_fast(self, traj_name: str, min_chunk_frames: int) -> bool:
        """Fast check if trajectory should be skipped (no I/O, no computation)."""
        # Check if completed (fastest check)
        if traj_name in self.progress_data["completed_trajectories"]:
            return True

        # Check if cached as insufficient frames with matching min_chunk_frames
        insufficient_cache = self.progress_data.get("insufficient_frame_trajectories", {})
        if traj_name in insufficient_cache:
            cached_min_frames = insufficient_cache[traj_name].get("min_chunk_frames")
            if cached_min_frames == min_chunk_frames:
                return True

        return False

    def mark_trajectory_completed(self, traj_name: str, chunk_ids: List[str], traj_data: Dict = None):
        """Mark trajectory as completed and save progress."""
        if traj_name not in self.progress_data["completed_trajectories"]:
            self.progress_data["completed_trajectories"].append(traj_name)

        # Remove from failed list if it was there
        if traj_name in self.progress_data["failed_trajectories"]:
            self.progress_data["failed_trajectories"].remove(traj_name)

        # Update chunk count
        self.progress_data["total_chunks"] += len(chunk_ids)

        # Store trajectory checksum
        if traj_data is not None:
            checksum = self._calculate_trajectory_checksum(traj_data)
            self.progress_data["trajectory_checksums"][traj_name] = checksum

        self._save_progress()

    def mark_trajectory_failed(self, traj_name: str, error_msg: str = ""):
        """Mark trajectory as failed."""
        if traj_name not in self.progress_data["failed_trajectories"]:
            self.progress_data["failed_trajectories"].append(traj_name)

        # Remove from completed list if it was there
        if traj_name in self.progress_data["completed_trajectories"]:
            self.progress_data["completed_trajectories"].remove(traj_name)

        print(f"Marked trajectory {traj_name} as failed: {error_msg}")
        self._save_progress()

    def mark_trajectory_incomplete(self, traj_name: str):
        """Mark trajectory as incomplete (remove from completed list)."""
        if traj_name in self.progress_data["completed_trajectories"]:
            self.progress_data["completed_trajectories"].remove(traj_name)
        if traj_name in self.progress_data["trajectory_checksums"]:
            del self.progress_data["trajectory_checksums"][traj_name]
        self._save_progress()

    def is_trajectory_insufficient_frames(self, traj_name: str, min_chunk_frames: int) -> bool:
        """Check if trajectory is cached as having insufficient frames."""
        insufficient_cache = self.progress_data.get("insufficient_frame_trajectories", {})
        if traj_name in insufficient_cache:
            cached_min_frames = insufficient_cache[traj_name].get("min_chunk_frames")
            # Only use cache if the minimum frame requirement hasn't changed
            if cached_min_frames == min_chunk_frames:
                return True
            else:
                # Remove from cache if requirement changed
                del insufficient_cache[traj_name]
                self._save_progress()
        return False

    def mark_trajectory_insufficient_frames(self, traj_name: str, frame_count: int, min_chunk_frames: int):
        """Cache trajectory as having insufficient frames."""
        if "insufficient_frame_trajectories" not in self.progress_data:
            self.progress_data["insufficient_frame_trajectories"] = {}

        self.progress_data["insufficient_frame_trajectories"][traj_name] = {
            "frame_count": frame_count,
            "min_chunk_frames": min_chunk_frames,
            "cached_time": time.time()
        }

        # Remove from other lists if present
        if traj_name in self.progress_data["completed_trajectories"]:
            self.progress_data["completed_trajectories"].remove(traj_name)
        if traj_name in self.progress_data["failed_trajectories"]:
            self.progress_data["failed_trajectories"].remove(traj_name)

        self._save_progress()

    def get_remaining_trajectories(self, all_trajectories: List[str], min_chunk_frames: int = None) -> List[str]:
        """Get list of trajectories that still need processing (optimized for minimal computation)."""
        # Use sets for O(1) lookup instead of O(n) list operations
        completed = set(self.progress_data["completed_trajectories"])
        insufficient = set()

        # If min_chunk_frames is provided, exclude cached insufficient frame trajectories
        if min_chunk_frames is not None:
            insufficient_cache = self.progress_data.get("insufficient_frame_trajectories", {})
            # Pre-filter insufficient trajectories with matching min_chunk_frames
            insufficient = {
                traj_name for traj_name, cache_data in insufficient_cache.items()
                if cache_data.get("min_chunk_frames") == min_chunk_frames
            }

        # Single set operation for exclusion (much faster than multiple checks)
        excluded = completed | insufficient
        return [traj for traj in all_trajectories if traj not in excluded]

    def get_progress_summary(self, total_trajectories: int) -> Dict:
        """Get summary of current progress."""
        completed = len(self.progress_data["completed_trajectories"])
        failed = len(self.progress_data["failed_trajectories"])
        insufficient_frames = len(self.progress_data.get("insufficient_frame_trajectories", {}))
        remaining = total_trajectories - completed - insufficient_frames

        elapsed_time = time.time() - self.progress_data["start_time"]

        return {
            "completed": completed,
            "failed": failed,
            "insufficient_frames": insufficient_frames,
            "remaining": remaining,
            "total": total_trajectories,
            "total_chunks": self.progress_data["total_chunks"],
            "elapsed_time": elapsed_time,
            "completion_rate": completed / total_trajectories if total_trajectories > 0 else 0
        }

    def set_processing_config(self, config: Dict):
        """Store processing configuration for validation on resume."""
        self.progress_data["processing_config"] = config
        self._save_progress()

    def validate_config(self, current_config: Dict) -> bool:
        """Validate that current config matches stored config."""
        stored_config = self.progress_data.get("processing_config", {})
        if not stored_config:
            return True  # No stored config, assume valid

        # Check critical parameters that would affect output
        critical_params = [
            "max_chunk_distance_m", "overlap_distance_m", "min_chunk_frames",
            "dino_model_type", "image_size"
        ]

        for param in critical_params:
            if stored_config.get(param) != current_config.get(param):
                print(f"Warning: Config parameter '{param}' changed from {stored_config.get(param)} to {current_config.get(param)}")
                return False

        return True


def process_trajectory(
    dataset: ViNT_Dataset,
    trajectory_name: str,
    feature_extractor: DiNOV2Extractor,
    max_chunk_distance_m: float,
    overlap_distance_m: float,
    min_chunk_frames: int,
    cache_dir: str,
    batch_size: int,
    device: str,
    fps_estimates: Optional[Dict] = None,
    dataset_name: str = "",
    progress_tracker: Optional[ProgressTracker] = None
) -> tuple:
    """
    Process a single trajectory, extract features, and create hybrid chunks.

    Args:
        dataset: ViNT_Dataset instance
        trajectory_name: Name of the trajectory
        feature_extractor: DINO feature extractor
        max_chunk_distance_m: Maximum distance per chunk in meters
        overlap_distance_m: Overlap distance between chunks in meters
        min_chunk_frames: Minimum number of frames for a chunk
        cache_dir: Path to save the feature cache
        batch_size: Batch size for feature extraction
        device: Device to use for feature extraction
        fps_estimates: FPS estimates from json file
        dataset_name: Name of dataset for FPS scaling
        progress_tracker: Optional progress tracker for resumable processing

    Returns:
        Tuple of (chunk_ids, feature_dim) or (None, None) if skipped
    """
    try:
        # Get trajectory data BEFORE scaling
        traj_data = dataset._get_trajectory(trajectory_name)
        assert "position" in traj_data and "yaw" in traj_data
        traj_len = len(traj_data["position"])
        images_sorted = sorted(
            f for f in os.listdir(trajectory_name)
            if not (f.endswith(".json") or f.endswith(".pkl"))
        )
        indices = list(range(len(images_sorted)))
        # Apply FPS scaling FIRST, before trajectory advancement check
        fps_scale_factor = calculate_fps_scaling_factor(dataset_name, fps_estimates)
        if fps_scale_factor > 1:
            original_traj_len = traj_len
            print(f"Applying FPS scaling for {dataset_name} dataset: keeping every {fps_scale_factor}th frame")
            traj_data = {
                "position": traj_data["position"][::fps_scale_factor],
                "yaw": traj_data["yaw"][::fps_scale_factor]
            }
            traj_len = len(traj_data["position"])
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

        # Create feature chunks using hybrid slicing (meters for max, frames for min)
        chunks = create_feature_chunks_meter_based(
            features,
            traj_data,
            indices,
            trajectory_name,
            cache_dir,
            max_chunk_distance_m,
            overlap_distance_m,
            min_chunk_frames
        )

        # Mark trajectory as completed in progress tracker
        if progress_tracker is not None:
            progress_tracker.mark_trajectory_completed(trajectory_name, chunks, traj_data)

        # Return the list of chunk IDs and feature dimension
        return chunks, features.shape[-1]

    except Exception as e:
        error_msg = f"Error processing trajectory {trajectory_name}: {str(e)}"
        print(error_msg)

        # Mark trajectory as failed in progress tracker
        if progress_tracker is not None:
            progress_tracker.mark_trajectory_failed(trajectory_name, error_msg)

        # Return None to indicate failure
        return None, None


def create_feature_chunks_meter_based(
    features: torch.Tensor,
    traj_data: List[Dict],
    valid_indices: List[int],
    trajectory_id: str,
    cache_dir: str,
    max_chunk_distance_m: float = 10.0,
    overlap_distance_m: float = 1.0,
    min_chunk_frames: int = 5
) -> List[str]:
    """
    Create feature chunks based on hybrid approach: meters for max, frames for min.

    Args:
        features: Tensor of features for a trajectory
        traj_data: Trajectory data containing positions and yaws
        valid_indices: List of valid indices in the original trajectory
        trajectory_id: ID of the trajectory
        cache_dir: Path to save the feature cache
        max_chunk_distance_m: Maximum distance per chunk (default 10m)
        overlap_distance_m: Overlap distance between chunks (default 1m)
        min_chunk_frames: Minimum number of frames for a chunk (default 5)

    Returns:
        List of chunk IDs
    """

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    positions = [pos[:2] for pos in traj_data["position"]]
    yaws = traj_data["yaw"]

    # Create hybrid chunks (meters for max, frames for min)
    chunk_ranges = create_meter_based_chunks(
        positions,
        max_chunk_distance_m,
        overlap_distance_m,
        min_chunk_frames
    )

    # Process each chunk
    CHUNKS = []
    for start_idx, end_idx in chunk_ranges:
        # Extract features and positions for this chunk
        chunk_features = features[start_idx:end_idx+1]  # Include end_idx
        chunk_positions = positions[start_idx:end_idx+1]
        chunk_yaws = yaws[start_idx:end_idx+1]
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
            "yaws":      chunk_yaws,                          # [global_frame_numbers]
            "indices":   chunk_indices                        # [global_frame_numbers]
        }

        # Save the features, positions, and trajectory data
        # Note: We're saving dictionaries with tensor values, which requires weights_only=False when loading
        torch.save(chunk_traj_data, chunk_path)

        #chunk_distance = calculate_distance_meters(
        #    chunk_positions[0][:2],
        #    chunk_positions[-1][:2]
        #)
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
    min_chunk_frames: int = 5,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dataset: ViNT_Dataset = None,
    keep_pt_files: bool = True,
    force_rebuild: bool = True,
    fps_estimates_path: str = None,
    dataset_name: str = "",
    enable_progress_tracking: bool = True,
):
    """
    Build a cache of DINO features from trajectory images using hybrid chunking.

    Args:
        dataset_dir: Path to the dataset directory
        cache_dir: Path to save the feature cache
        feature_extractor: DINO feature extractor instance
        max_chunk_distance_m: Maximum distance per chunk in meters (10m)
        overlap_distance_m: Overlap distance between chunks in meters (1m)
        min_chunk_frames: Minimum number of frames for a chunk (5 frames)
        batch_size: Batch size for feature extraction
        device: Device to use for feature extraction
        dataset: ViNT_Dataset instance (if already created)
        keep_pt_files: Whether to keep the .pt files after creating the LMDB cache
        force_rebuild: Whether to force rebuilding the cache even if it already exists
        fps_estimates_path: Path to fps_estimates.json file
        dataset_name: Name of dataset for FPS scaling
        enable_progress_tracking: Whether to enable progress tracking for resumable processing
    """
    # Load FPS estimates
    fps_estimates = None
    if fps_estimates_path and os.path.exists(fps_estimates_path):
        with open(fps_estimates_path, 'r') as f:
            fps_estimates = json.load(f)
        print(f"Loaded FPS estimates for {len(fps_estimates)} datasets")

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Initialize progress tracker
    progress_tracker = None
    if enable_progress_tracking:
        progress_tracker = ProgressTracker(cache_dir, dataset_name)

        # Store current processing configuration
        current_config = {
            "max_chunk_distance_m": max_chunk_distance_m,
            "overlap_distance_m": overlap_distance_m,
            "min_chunk_frames": min_chunk_frames,
            "batch_size": batch_size,
            "dataset_name": dataset_name
        }

        # Validate configuration if resuming
        if not force_rebuild and not progress_tracker.validate_config(current_config):
            print("Configuration has changed since last run. Use force_rebuild=True to start fresh.")
            response = input("Continue with current config? (y/N): ").lower().strip()
            if response != 'y':
                print("Aborting. Use force_rebuild=True to start with new configuration.")
                return

        progress_tracker.set_processing_config(current_config)
        print(f"Progress tracking enabled. Progress file: {progress_tracker.progress_file}")

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
        # Get list of trajectories to process (skip completed ones and cached insufficient frame ones if resuming)
        trajectories_to_process = dataset.traj_names
        if progress_tracker and not force_rebuild:
            trajectories_to_process = progress_tracker.get_remaining_trajectories(dataset.traj_names, min_chunk_frames)
            if len(trajectories_to_process) < len(dataset.traj_names):
                progress_summary = progress_tracker.get_progress_summary(len(dataset.traj_names))
                print(f"Resuming processing: {progress_summary['completed']}/{progress_summary['total']} trajectories completed")
                if progress_summary['insufficient_frames'] > 0:
                    print(f"Cached insufficient frame trajectories: {progress_summary['insufficient_frames']}")
                print(f"Remaining trajectories: {len(trajectories_to_process)}")

        # Process trajectories with progress tracking
        for traj_name in tqdm(trajectories_to_process, desc="Processing trajectories"):
            # Ultra-fast check: skip if completed or cached as insufficient (no I/O, minimal computation)
            if progress_tracker and progress_tracker.should_skip_trajectory_fast(traj_name, min_chunk_frames):
                continue

            # Only load trajectory data if we need to process it
            traj_data = dataset._get_trajectory(traj_name)
            if traj_data is None:
                print(f"Trajectory data not found for {traj_name}")
                if progress_tracker:
                    progress_tracker.mark_trajectory_failed(traj_name, "Trajectory data not found")
                continue

            # Check if already completed (with data validation) - only for force_rebuild or data changes
            if progress_tracker and not force_rebuild:
                if progress_tracker.is_trajectory_completed(traj_name, traj_data):
                    print(f"Trajectory {traj_name} already completed, skipping")
                    continue

            # Check trajectory length in frames for minimum requirement
            trajectory_length = len(traj_data["position"])
            if trajectory_length < min_chunk_frames:
                print(f"Trajectory {traj_name} has {trajectory_length} frames (< {min_chunk_frames} frames), caching and skipping.")
                if progress_tracker:
                    progress_tracker.mark_trajectory_insufficient_frames(traj_name, trajectory_length, min_chunk_frames)
                continue

            # Process the trajectory with hybrid parameters and FPS scaling
            chunk_ids, traj_feature_dim = process_trajectory(
                dataset,
                traj_name,
                feature_extractor,
                max_chunk_distance_m,
                overlap_distance_m,
                min_chunk_frames,
                cache_dir,
                batch_size,
                device,
                fps_estimates,
                dataset_name,
                progress_tracker
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

            # Print progress summary periodically
            if progress_tracker and len(progress_tracker.progress_data["completed_trajectories"]) % 10 == 0:
                progress_summary = progress_tracker.get_progress_summary(len(dataset.traj_names))
                elapsed_hours = progress_summary["elapsed_time"] / 3600
                print(f"Progress: {progress_summary['completed']}/{progress_summary['total']} trajectories "
                      f"({progress_summary['completion_rate']:.1%}), "
                      f"{progress_summary['total_chunks']} chunks, "
                      f"{elapsed_hours:.1f}h elapsed")

    # Write the feature dimension into the metadata
    
    if len(CHUNKS) > 0:
    
        if feature_dim is None:                        # warm-start path
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
            "min_chunk_frames": min_chunk_frames,
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
    
    else:
        print(f"Warning: No chunks were created for {dataset_name}. Check your parameters and data.")

    # Print final progress summary
    if progress_tracker:
        final_summary = progress_tracker.get_progress_summary(len(dataset.traj_names))
        elapsed_hours = final_summary["elapsed_time"] / 3600
        print(f"\n📊 Final Progress Summary:")
        print(f"  ✅ Completed: {final_summary['completed']}/{final_summary['total']} trajectories ({final_summary['completion_rate']:.1%})")
        print(f"  ❌ Failed: {final_summary['failed']} trajectories")
        print(f"  📏 Insufficient frames: {final_summary['insufficient_frames']} trajectories")
        print(f"  📦 Total chunks: {final_summary['total_chunks']}")
        print(f"  ⏱️  Total time: {elapsed_hours:.1f} hours")

        if final_summary['failed'] > 0:
            print(f"  ⚠️  Failed trajectories: {progress_tracker.progress_data['failed_trajectories']}")

        if final_summary['insufficient_frames'] > 0:
            insufficient_trajs = list(progress_tracker.progress_data.get('insufficient_frame_trajectories', {}).keys())
            print(f"  📏 Insufficient frame trajectories: {insufficient_trajs[:5]}{'...' if len(insufficient_trajs) > 5 else ''}")

    # Create LMDB environment for faster access (only if it doesn't exist or force rebuild)
    #lmdb_path = os.path.join(cache_dir, "lmdb_cache")
    #if not os.path.exists(lmdb_path) or force_rebuild:
    #    print("Building LMDB cache...")
    #    create_lmdb_cache(cache_dir, cache_metadata, keep_pt_files=keep_pt_files)
    #else:
    #    print(f"LMDB cache already exists at {lmdb_path}, skipping creation")



def main(config, data_configs):
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
                    min_goal_distance_meters=data_configs.get("min_goal_distance_meters",1.0),
                    max_goal_distance_meters=data_configs.get("max_goal_distance_meters",10.0),
                    negative_mining        = viz_cfg.get("negative_mining",True),
                    len_traj_pred          = config["len_traj_pred"],
                    context_size           = config["context_size"],
                    end_slack              = viz_cfg.get("end_slack",0),
                    normalize               = config["normalize"],
                )
                build_dino_feature_cache(
                    dataset_dir                = folder,
                    cache_dir                  = cache_dir,
                    feature_extractor          = feature_extractor,
                    max_chunk_distance_m       = config["max_chunk_distance_m"],
                    overlap_distance_m         = config["overlap_distance_m"],
                    min_chunk_frames           = config["min_chunk_frames"],
                    batch_size                 = config["batch_size"],
                    device                     = device,
                    dataset                    = viz_ds,
                    keep_pt_files              = config["keep_pt_files"],
                    fps_estimates_path         = "/home/ubuntu/SatiNav/training_server/Sati_data/fps_estimates.json",
                    dataset_name               = f"viz_{split}",
                    enable_progress_tracking   = config.get("enable_progress_tracking", True),
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
                min_goal_distance_meters=data_configs["min_goal_distance_meters"],
                max_goal_distance_meters=data_configs["max_goal_distance_meters"],
                negative_mining        = data_config["negative_mining"],
                len_traj_pred          = config["len_traj_pred"],
                context_size           = config["context_size"],
                end_slack              = data_config["end_slack"],
                normalize               = config["normalize"],
                force_rebuild_indices  = data_configs.get("force_rebuild_indices", False),
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
                min_chunk_frames           = config["min_chunk_frames"],
                batch_size                 = config["batch_size"],
                device                     = device,
                dataset                    = ds,
                keep_pt_files              = config["keep_pt_files"],
                fps_estimates_path         = "/home/ubuntu/SatiNav/training_server/Sati_data/fps_estimates.json",
                dataset_name               = dataset_name,
                enable_progress_tracking   = config.get("enable_progress_tracking", True),
            )

    print("✅ DINO feature cache built for all datasets & viz_data!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINO Feature Cache Builder")
    parser.add_argument(
        "--config",
        "-c",
        default="/app/visualnav-transformer/config/data/dino_cache.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    # Load user config
    with open(os.path.join(ROOT_DIR, args.config), "r") as f:
        config = yaml.safe_load(f)
    
    # Load data config
    with open(os.path.join(ROOT_DIR, "config/data/data_config.yaml"), "r") as f:
        data_configs = yaml.safe_load(f)

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    main(config, data_configs)