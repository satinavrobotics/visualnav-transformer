import numpy as np
import torch
import os

def calculate_fps_scaling_factor(dataset_name, fps_estimates, max_scaling_factor=100):
    """
    Calculate FPS scaling factor based on fps_estimates.json data.
    
    For datasets like Etna with extremely small median displacement,
    we scale them by keeping every X-th frame to match the minimum median 
    displacement of other correctly sampled datasets (less aggressive scaling).
    
    Args:
        dataset_name: Name of the dataset
        fps_estimates: Dictionary from fps_estimates.json
        
    Returns:
        scale_factor: Keep every N-th frame (1 = no scaling)
    """
    if not fps_estimates or dataset_name not in fps_estimates:
        return 1
    
    current_median_disp = fps_estimates[dataset_name]["median_disp_m"]
    all_median_disps = [d["median_disp_m"] for d in fps_estimates.values()]
    if len(all_median_disps) < 2:
        # Fallback: if no other datasets available, no scaling
        return 1
    
    q1 = np.percentile(all_median_disps, 25)
    q3 = np.percentile(all_median_disps, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
 
    if current_median_disp > upper_bound:
        print(f"Warning: Dataset {dataset_name} has very high displacement ({current_median_disp:.8f}m)")
        return 1
    elif current_median_disp < lower_bound:
        target_disp = lower_bound
    else:
        target_disp = current_median_disp
    
    # Add numerical stability check
    if current_median_disp <= 1e-8:  # Effectively zero
        print(f"Warning: Dataset {dataset_name} has near-zero displacement ({current_median_disp:.8f}m), using max scaling factor")
        return max_scaling_factor  # Conservative max scaling
    
    # Only scale if current median is below our target
    ratio = target_disp / current_median_disp
    if ratio > 1.0:
        scale_factor = min(max_scaling_factor, int(np.ceil(ratio)))
    else:
        scale_factor = 1
    
    if scale_factor > 1:
        actual_scaled_disp = current_median_disp * scale_factor
        print(f"Dataset {dataset_name}: median_disp={current_median_disp:.6f}m, target≤{target_disp:.6f}m → scaling ×{scale_factor}")
        print(f"  After scaling: {actual_scaled_disp:.6f}m per frame")
 
    return scale_factor


def calculate_distance_meters(pos1, pos2):
    """Calculate Euclidean distance between two positions in meters."""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def calculate_trajectory_advancement(positions):
    """Calculate total advancement (distance traveled) in a trajectory."""
    if len(positions) < 2:
        return 0.0

    total_distance = 0.0
    for i in range(1, len(positions)):
        total_distance += calculate_distance_meters(positions[i-1], positions[i])

    return total_distance


def setup_gpu(config):
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
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu")
    return device



def create_meter_based_chunks(positions, max_chunk_distance_m=10.0, overlap_distance_m=1.0, min_chunk_frames=5):
    """
    Create chunks based on meter-based advancement with frame-based minimum.

    Args:
        positions: List of [x, y] positions
        max_chunk_distance_m: Maximum distance per chunk (default 10m)
        overlap_distance_m: Overlap distance between chunks (default 1m)
        min_chunk_frames: Minimum number of frames for a chunk (default 5)

    Returns:
        List of (start_idx, end_idx) tuples for each chunk
    """
    if len(positions) < 2:
        return []

    chunks = []
    start_idx = 0
    while start_idx < len(positions) - 1:
        current_distance = 0.0
        # Find end index for this chunk (max 10m advancement)
        end_idx = start_idx + 1
        for i in range(start_idx + 1, len(positions)):
            segment_distance = calculate_distance_meters(positions[i-1], positions[i])
            if current_distance + segment_distance >= max_chunk_distance_m:
                break
            current_distance += segment_distance
            end_idx = i

        # Check if this would be the last chunk and it's too small (in frames)
        remaining_frames = len(positions) - 1 - end_idx

        # If remaining frames is less than min_chunk_frames, extend current chunk
        if remaining_frames < min_chunk_frames and end_idx < len(positions) - 1:
            end_idx = len(positions) - 1

        chunks.append((start_idx, end_idx))

        # Calculate next start index with overlap
        if end_idx >= len(positions) - 1:
            break

        # Find start of next chunk (with overlap_distance_m overlap)
        # walk backwards accumulating overlap until we exceed overlap_distance_m
        overlap_distance = 0.0
        next_start_idx = end_idx
        i = end_idx
        while i > start_idx and overlap_distance <= overlap_distance_m:
            segment_distance = calculate_distance_meters(positions[i], positions[i-1])
            overlap_distance += segment_distance
            next_start_idx = i - 1
            i -= 1
            if overlap_distance + segment_distance <= overlap_distance_m:
                overlap_distance += segment_distance
                next_start_idx = i - 1
            else:
                break

        start_idx = max(next_start_idx, start_idx + 1)  # Ensure progress

    return chunks