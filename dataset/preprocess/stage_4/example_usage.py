#!/usr/bin/env python3
"""
Example usage of DINO cache preprocessing with progress tracking.
This script demonstrates how to use the enhanced DINO cache preprocessor
with robust progress tracking capabilities.
"""

import os
import yaml
import argparse

def run_dino_cache_with_progress():
    """
    Example of running DINO cache preprocessing with progress tracking enabled.
    """
    
    # Example configuration
    config = {
        "project_name": "example_dino_cache",
        "run_name": "dino_cache_with_progress",
        
        # DINO cache parameters
        "dino_model_type": "large",
        "max_chunk_distance_m": 50.0,
        "overlap_distance_m": 1.0,
        "min_chunk_distance_m": 0.3,
        "batch_size": 32,
        "keep_pt_files": True,
        
        # Progress tracking (NEW!)
        "enable_progress_tracking": True,
        
        # GPU settings
        "gpu_ids": [0],
        "num_workers": 8,
        "seed": 0,
        
        # Normalization and context
        "normalize": True,
        "context_size": 3,
        "len_traj_pred": 8,
        
        # Dataset parameters
        "image_size": [320, 240]
    }
    
    # Example data configuration
    data_configs = {
        "min_goal_distance_meters": 1.0,
        "max_goal_distance_meters": 10.0,
        "force_rebuild_indices": False,
        "datasets": {
            "example_dataset": {
                "data_folder": "/path/to/your/dataset",
                "available": True,
                "negative_mining": True,
                "goals_per_obs": 1,
                "end_slack": 0,
                "waypoint_spacing": 1,
                "metric_waypoint_spacing": 1.0
            }
        }
    }
    
    print("🚀 Starting DINO cache preprocessing with progress tracking...")
    print("📋 Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Import and run the main function
    # Note: In practice, you would run this via command line or import the module
    print("\n💡 To run this configuration, save it to a YAML file and use:")
    print("python dino_cache.py --config your_config.yaml")
    
    return config, data_configs

def demonstrate_progress_management():
    """
    Demonstrate progress management utilities.
    """
    print("\n📊 Progress Management Examples:")
    print("=" * 50)
    
    cache_dir = "/path/to/your/cache/dino_cache_large"
    
    print(f"1. Show progress for cache directory:")
    print(f"   python progress_utils.py show --cache-dir {cache_dir}")
    
    print(f"\n2. List all cache directories with progress:")
    print(f"   python progress_utils.py list")
    
    print(f"\n3. Reset failed trajectories to retry them:")
    print(f"   python progress_utils.py reset-failed --cache-dir {cache_dir}")
    
    print(f"\n4. Reset all progress (start fresh):")
    print(f"   python progress_utils.py reset --cache-dir {cache_dir}")
    
    print(f"\n5. Reset without confirmation prompts:")
    print(f"   python progress_utils.py reset --cache-dir {cache_dir} --yes")

def demonstrate_resumable_workflow():
    """
    Demonstrate a typical resumable workflow.
    """
    print("\n🔄 Resumable Workflow Example:")
    print("=" * 50)
    
    workflow_steps = [
        "1. Start processing: python dino_cache.py --config config.yaml",
        "2. Processing runs for several hours...",
        "3. ❌ Process interrupted (power outage, system restart, etc.)",
        "4. ✅ Restart: python dino_cache.py --config config.yaml",
        "5. 📊 System automatically detects existing progress",
        "6. 🔍 Validates configuration hasn't changed",
        "7. ▶️  Resumes from last completed trajectory",
        "8. 🎉 Processing completes successfully"
    ]
    
    for step in workflow_steps:
        print(f"   {step}")
    
    print("\n💡 Key Benefits:")
    benefits = [
        "✅ No lost work from interruptions",
        "✅ Automatic detection of completed trajectories", 
        "✅ Configuration validation prevents inconsistencies",
        "✅ Individual trajectory failures don't stop entire process",
        "✅ Easy retry of failed trajectories",
        "✅ Real-time progress monitoring"
    ]
    
    for benefit in benefits:
        print(f"   {benefit}")

def demonstrate_error_handling():
    """
    Demonstrate error handling and recovery.
    """
    print("\n🛠️  Error Handling & Recovery:")
    print("=" * 50)
    
    scenarios = [
        {
            "scenario": "Individual Trajectory Failure",
            "description": "One trajectory fails due to corrupted data",
            "handling": [
                "❌ Trajectory marked as failed",
                "📝 Error message logged",
                "▶️  Processing continues with next trajectory",
                "🔄 Failed trajectory can be retried later"
            ]
        },
        {
            "scenario": "Configuration Change",
            "description": "User changes chunk size parameters",
            "handling": [
                "⚠️  System detects configuration change",
                "❓ Prompts user for confirmation",
                "🔄 Option to reset progress and start fresh",
                "✅ Prevents inconsistent output"
            ]
        },
        {
            "scenario": "Data Change Detection",
            "description": "Trajectory data modified since last run",
            "handling": [
                "🔍 Checksum validation detects change",
                "📝 Trajectory marked for reprocessing",
                "🔄 Automatically reprocessed on next run",
                "✅ Ensures output consistency"
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 {scenario['scenario']}:")
        print(f"   Description: {scenario['description']}")
        print(f"   Handling:")
        for step in scenario['handling']:
            print(f"     {step}")

def main():
    """
    Main function demonstrating DINO cache preprocessing with progress tracking.
    """
    parser = argparse.ArgumentParser(description="DINO Cache Progress Tracking Example")
    parser.add_argument("--demo", choices=["config", "progress", "workflow", "errors", "all"], 
                       default="all", help="Which demonstration to run")
    
    args = parser.parse_args()
    
    print("🎯 DINO Cache Preprocessing with Progress Tracking")
    print("=" * 60)
    
    if args.demo in ["config", "all"]:
        config, data_configs = run_dino_cache_with_progress()
    
    if args.demo in ["progress", "all"]:
        demonstrate_progress_management()
    
    if args.demo in ["workflow", "all"]:
        demonstrate_resumable_workflow()
    
    if args.demo in ["errors", "all"]:
        demonstrate_error_handling()
    
    print("\n" + "=" * 60)
    print("📚 For more information, see README_progress_tracking.md")
    print("🧪 Run tests with: python test_progress_tracking.py")
    print("🔧 Manage progress with: python progress_utils.py --help")

if __name__ == "__main__":
    main()
