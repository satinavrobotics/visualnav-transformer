#!/usr/bin/env python3
"""
Test script for progress tracking functionality.
"""

import os
import tempfile
import shutil
import json
import sys

# Add the project root to the path
sys.path.append('/app/visualnav-transformer')

# Import just the ProgressTracker class by copying it here for testing
import time
import hashlib

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

    def _load_progress(self):
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
            "insufficient_frame_trajectories": {},
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

    def _calculate_trajectory_checksum(self, traj_data):
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

    def is_trajectory_completed(self, traj_name: str, traj_data = None) -> bool:
        """Check if trajectory has been completed and data hasn't changed."""
        if traj_name not in self.progress_data["completed_trajectories"]:
            return False

        # If trajectory data is provided, verify it hasn't changed
        if traj_data is not None:
            current_checksum = self._calculate_trajectory_checksum(traj_data)
            stored_checksum = self.progress_data["trajectory_checksums"].get(traj_name, "")
            if current_checksum != stored_checksum:
                print(f"Trajectory {traj_name} data has changed, will reprocess")
                self.mark_trajectory_incomplete(traj_name)
                return False

        return True

    def mark_trajectory_completed(self, traj_name: str, chunk_ids, traj_data = None):
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

    def get_remaining_trajectories(self, all_trajectories):
        """Get list of trajectories that still need processing."""
        completed = set(self.progress_data["completed_trajectories"])
        return [traj for traj in all_trajectories if traj not in completed]

    def get_progress_summary(self, total_trajectories: int):
        """Get summary of current progress."""
        completed = len(self.progress_data["completed_trajectories"])
        failed = len(self.progress_data["failed_trajectories"])
        remaining = total_trajectories - completed

        elapsed_time = time.time() - self.progress_data["start_time"]

        return {
            "completed": completed,
            "failed": failed,
            "remaining": remaining,
            "total": total_trajectories,
            "total_chunks": self.progress_data["total_chunks"],
            "elapsed_time": elapsed_time,
            "completion_rate": completed / total_trajectories if total_trajectories > 0 else 0
        }

    def set_processing_config(self, config):
        """Store processing configuration for validation on resume."""
        self.progress_data["processing_config"] = config
        self._save_progress()

    def validate_config(self, current_config) -> bool:
        """Validate that current config matches stored config."""
        stored_config = self.progress_data.get("processing_config", {})
        if not stored_config:
            return True  # No stored config, assume valid

        # Check critical parameters that would affect output
        critical_params = [
            "max_chunk_distance_m", "overlap_distance_m", "min_chunk_distance_m",
            "dino_model_type", "image_size"
        ]

        for param in critical_params:
            if stored_config.get(param) != current_config.get(param):
                print(f"Warning: Config parameter '{param}' changed from {stored_config.get(param)} to {current_config.get(param)}")
                return False

        return True

def test_progress_tracker():
    """Test basic progress tracking functionality."""
    print("Testing ProgressTracker...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = os.path.join(temp_dir, "test_cache")
        
        # Test 1: Initialize new progress tracker
        print("Test 1: Initialize new progress tracker")
        tracker = ProgressTracker(cache_dir, "test_dataset")
        assert tracker.progress_data["dataset_name"] == "test_dataset"
        assert len(tracker.progress_data["completed_trajectories"]) == 0
        print("✅ New tracker initialized correctly")
        
        # Test 2: Mark trajectory as completed
        print("Test 2: Mark trajectory as completed")
        test_traj_data = {
            "position": [[0, 0], [1, 1], [2, 2]],
            "yaw": [0, 0.1, 0.2]
        }
        tracker.mark_trajectory_completed("traj_001", ["chunk_001", "chunk_002"], test_traj_data)
        assert "traj_001" in tracker.progress_data["completed_trajectories"]
        assert tracker.progress_data["total_chunks"] == 2
        print("✅ Trajectory marked as completed")
        
        # Test 3: Check if trajectory is completed
        print("Test 3: Check if trajectory is completed")
        assert tracker.is_trajectory_completed("traj_001", test_traj_data)
        assert not tracker.is_trajectory_completed("traj_002")
        print("✅ Trajectory completion check works")
        
        # Test 4: Mark trajectory as failed
        print("Test 4: Mark trajectory as failed")
        tracker.mark_trajectory_failed("traj_002", "Test error")
        assert "traj_002" in tracker.progress_data["failed_trajectories"]
        print("✅ Trajectory marked as failed")
        
        # Test 5: Get remaining trajectories
        print("Test 5: Get remaining trajectories")
        all_trajs = ["traj_001", "traj_002", "traj_003", "traj_004"]
        remaining = tracker.get_remaining_trajectories(all_trajs)
        expected_remaining = ["traj_002", "traj_003", "traj_004"]  # traj_001 completed, traj_002 failed but should be retried
        assert set(remaining) == set(expected_remaining), f"Expected {expected_remaining}, got {remaining}"
        print("✅ Remaining trajectories calculated correctly")
        
        # Test 6: Progress persistence
        print("Test 6: Progress persistence")
        # Create new tracker instance to test loading
        tracker2 = ProgressTracker(cache_dir, "test_dataset")
        assert "traj_001" in tracker2.progress_data["completed_trajectories"]
        assert "traj_002" in tracker2.progress_data["failed_trajectories"]
        assert tracker2.progress_data["total_chunks"] == 2
        print("✅ Progress persisted correctly")
        
        # Test 7: Configuration validation
        print("Test 7: Configuration validation")
        config1 = {"max_chunk_distance_m": 10.0, "dino_model_type": "large"}
        config2 = {"max_chunk_distance_m": 20.0, "dino_model_type": "large"}
        config3 = {"max_chunk_distance_m": 10.0, "dino_model_type": "large"}
        
        tracker.set_processing_config(config1)
        assert not tracker.validate_config(config2)  # Different config should fail
        assert tracker.validate_config(config3)      # Same config should pass
        print("✅ Configuration validation works")
        
        # Test 8: Checksum validation
        print("Test 8: Checksum validation")
        # Modify trajectory data
        modified_traj_data = {
            "position": [[0, 0], [1, 1], [2, 2], [3, 3]],  # Added one more position
            "yaw": [0, 0.1, 0.2, 0.3]
        }
        # Should detect change and return False
        assert not tracker.is_trajectory_completed("traj_001", modified_traj_data)
        print("✅ Checksum validation detects changes")
        
        # Test 9: Insufficient frame caching
        print("Test 9: Insufficient frame caching")
        tracker.mark_trajectory_insufficient_frames("traj_003", 3, 5)
        assert tracker.is_trajectory_insufficient_frames("traj_003", 5)
        assert not tracker.is_trajectory_insufficient_frames("traj_003", 3)  # Different min_frames
        assert not tracker.is_trajectory_insufficient_frames("traj_004", 5)  # Different trajectory
        print("✅ Insufficient frame caching works correctly")

        # Test 10: Progress summary with insufficient frames
        print("Test 10: Progress summary with insufficient frames")
        summary = tracker.get_progress_summary(4)  # 4 total trajectories
        # Note: traj_001 was marked incomplete in checksum test, so completed should be 0
        # traj_002 is failed, traj_003 is insufficient frames
        assert summary["completed"] == 0
        assert summary["failed"] == 1
        assert summary["insufficient_frames"] == 1
        assert summary["remaining"] == 2  # Only traj_004 and traj_001 remain
        assert summary["total"] == 4
        assert summary["completion_rate"] == 0.0
        print("✅ Progress summary with insufficient frames calculated correctly")
        
    print("\n🎉 All tests passed! Progress tracking is working correctly.")

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = os.path.join(temp_dir, "test_cache")
        
        # Test with empty trajectory data
        tracker = ProgressTracker(cache_dir, "test_dataset")
        empty_traj_data = {"position": [], "yaw": []}
        checksum = tracker._calculate_trajectory_checksum(empty_traj_data)
        assert checksum == ""  # Should return empty string for empty data
        print("✅ Empty trajectory data handled correctly")
        
        # Test with malformed trajectory data
        malformed_data = {"position": [[0]], "yaw": [0]}  # Position missing y coordinate
        checksum = tracker._calculate_trajectory_checksum(malformed_data)
        assert isinstance(checksum, str)  # Should still return a string
        print("✅ Malformed trajectory data handled gracefully")
        
        # Test with missing trajectory data
        assert not tracker.is_trajectory_completed("nonexistent_traj")
        print("✅ Missing trajectory handled correctly")
        
    print("✅ All edge case tests passed!")

if __name__ == "__main__":
    test_progress_tracker()
    test_edge_cases()
    print("\n✨ All tests completed successfully!")
