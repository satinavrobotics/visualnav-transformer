#!/usr/bin/env python3
"""
Utility script for managing DINO cache preprocessing progress.
Provides commands to view, reset, and manage progress tracking.
"""

import argparse
import os
import json
import sys
from typing import Dict, List

def load_progress(cache_dir: str) -> Dict:
    """Load progress data from cache directory."""
    progress_file = os.path.join(cache_dir, "progress.json")
    if not os.path.exists(progress_file):
        print(f"No progress file found at {progress_file}")
        return {}
    
    try:
        with open(progress_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading progress file: {e}")
        return {}

def show_progress(cache_dir: str):
    """Show current progress status."""
    progress_data = load_progress(cache_dir)
    if not progress_data:
        return
    
    print(f"\n📊 Progress Status for {cache_dir}")
    print("=" * 60)
    
    # Basic stats
    completed = len(progress_data.get("completed_trajectories", []))
    failed = len(progress_data.get("failed_trajectories", []))
    insufficient_frames = len(progress_data.get("insufficient_frame_trajectories", {}))
    total_chunks = progress_data.get("total_chunks", 0)

    print(f"Dataset: {progress_data.get('dataset_name', 'Unknown')}")
    print(f"Completed trajectories: {completed}")
    print(f"Failed trajectories: {failed}")
    print(f"Insufficient frame trajectories: {insufficient_frames}")
    print(f"Total chunks generated: {total_chunks}")
    
    # Timing info
    if "start_time" in progress_data and "last_update" in progress_data:
        import time
        start_time = progress_data["start_time"]
        last_update = progress_data["last_update"]
        elapsed = last_update - start_time
        elapsed_hours = elapsed / 3600
        print(f"Processing time: {elapsed_hours:.1f} hours")
        print(f"Last update: {time.ctime(last_update)}")
    
    # Configuration
    config = progress_data.get("processing_config", {})
    if config:
        print(f"\nProcessing Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # Failed trajectories
    failed_trajs = progress_data.get("failed_trajectories", [])
    if failed_trajs:
        print(f"\n❌ Failed Trajectories ({len(failed_trajs)}):")
        for traj in failed_trajs[:10]:  # Show first 10
            print(f"  - {traj}")
        if len(failed_trajs) > 10:
            print(f"  ... and {len(failed_trajs) - 10} more")

    # Insufficient frame trajectories
    insufficient_trajs = progress_data.get("insufficient_frame_trajectories", {})
    if insufficient_trajs:
        print(f"\n📏 Insufficient Frame Trajectories ({len(insufficient_trajs)}):")
        for traj, info in list(insufficient_trajs.items())[:10]:  # Show first 10
            frame_count = info.get("frame_count", "unknown")
            min_required = info.get("min_chunk_frames", "unknown")
            print(f"  - {traj}: {frame_count} frames (< {min_required} required)")
        if len(insufficient_trajs) > 10:
            print(f"  ... and {len(insufficient_trajs) - 10} more")

def reset_progress(cache_dir: str, confirm: bool = False):
    """Reset progress tracking."""
    progress_file = os.path.join(cache_dir, "progress.json")
    
    if not os.path.exists(progress_file):
        print(f"No progress file found at {progress_file}")
        return
    
    if not confirm:
        response = input(f"Are you sure you want to reset progress for {cache_dir}? (y/N): ")
        if response.lower().strip() != 'y':
            print("Cancelled.")
            return
    
    try:
        os.remove(progress_file)
        print(f"Progress file removed: {progress_file}")
    except OSError as e:
        print(f"Error removing progress file: {e}")

def reset_failed(cache_dir: str, confirm: bool = False):
    """Reset only failed trajectories to retry them."""
    progress_data = load_progress(cache_dir)
    if not progress_data:
        return
    
    failed_trajs = progress_data.get("failed_trajectories", [])
    if not failed_trajs:
        print("No failed trajectories to reset.")
        return
    
    if not confirm:
        print(f"Found {len(failed_trajs)} failed trajectories:")
        for traj in failed_trajs[:5]:
            print(f"  - {traj}")
        if len(failed_trajs) > 5:
            print(f"  ... and {len(failed_trajs) - 5} more")
        
        response = input(f"Reset {len(failed_trajs)} failed trajectories? (y/N): ")
        if response.lower().strip() != 'y':
            print("Cancelled.")
            return
    
    # Clear failed trajectories
    progress_data["failed_trajectories"] = []
    
    # Save updated progress
    progress_file = os.path.join(cache_dir, "progress.json")
    try:
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        print(f"Reset {len(failed_trajs)} failed trajectories. They will be retried on next run.")
    except IOError as e:
        print(f"Error saving progress file: {e}")

def list_cache_dirs(base_dir: str = "."):
    """List all cache directories with progress files."""
    cache_dirs = []
    
    for root, dirs, files in os.walk(base_dir):
        if "progress.json" in files and "dino_cache" in root:
            cache_dirs.append(root)
    
    if not cache_dirs:
        print("No cache directories with progress files found.")
        return
    
    print(f"Found {len(cache_dirs)} cache directories with progress:")
    for cache_dir in sorted(cache_dirs):
        progress_data = load_progress(cache_dir)
        completed = len(progress_data.get("completed_trajectories", []))
        failed = len(progress_data.get("failed_trajectories", []))
        insufficient = len(progress_data.get("insufficient_frame_trajectories", {}))
        dataset_name = progress_data.get("dataset_name", "Unknown")
        print(f"  {cache_dir} ({dataset_name}): {completed} completed, {failed} failed, {insufficient} insufficient frames")

def main():
    parser = argparse.ArgumentParser(description="DINO Cache Progress Management Utility")
    parser.add_argument("command", choices=["show", "reset", "reset-failed", "list"], 
                       help="Command to execute")
    parser.add_argument("--cache-dir", "-d", type=str, 
                       help="Path to cache directory (required for show, reset, reset-failed)")
    parser.add_argument("--yes", "-y", action="store_true", 
                       help="Skip confirmation prompts")
    parser.add_argument("--base-dir", "-b", type=str, default=".", 
                       help="Base directory to search for cache dirs (for list command)")
    
    args = parser.parse_args()
    
    if args.command in ["show", "reset", "reset-failed"] and not args.cache_dir:
        print(f"Error: --cache-dir is required for '{args.command}' command")
        sys.exit(1)
    
    if args.command == "show":
        show_progress(args.cache_dir)
    elif args.command == "reset":
        reset_progress(args.cache_dir, args.yes)
    elif args.command == "reset-failed":
        reset_failed(args.cache_dir, args.yes)
    elif args.command == "list":
        list_cache_dirs(args.base_dir)

if __name__ == "__main__":
    main()
