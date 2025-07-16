# DINO Cache Progress Tracking

This document describes the robust progress tracking mechanism added to the DINO cache preprocessor, which allows for reliable and resumable processing of large datasets.

## Features

### 🔄 Resumable Processing
- **Automatic Resume**: If processing is interrupted, the system can resume from where it left off
- **Trajectory-Level Tracking**: Tracks completion status for each trajectory individually
- **Chunk Counting**: Keeps track of total chunks generated across all trajectories

### 🛡️ Data Integrity
- **Checksum Validation**: Detects if trajectory data has changed since last processing
- **Configuration Validation**: Ensures processing parameters haven't changed between runs
- **Atomic Progress Saves**: Uses atomic file operations to prevent corruption

### 📊 Progress Monitoring
- **Real-time Progress**: Shows completion percentage and estimated time remaining
- **Failed Trajectory Tracking**: Keeps track of trajectories that failed processing
- **Detailed Statistics**: Provides comprehensive progress summaries

### 🔧 Error Handling
- **Graceful Failure**: Individual trajectory failures don't stop the entire process
- **Error Logging**: Detailed error messages for failed trajectories
- **Retry Mechanism**: Failed trajectories can be easily retried

## Usage

### Basic Usage

The progress tracking is enabled by default. Simply run the DINO cache preprocessor as usual:

```bash
python dino_cache.py --config config/data/dino_cache.yaml
```

### Configuration

Add to your `dino_cache.yaml`:

```yaml
enable_progress_tracking: True  # Enable progress tracking (default: True)
```

### Progress Management

Use the `progress_utils.py` script to manage progress:

```bash
# Show progress for a specific cache directory
python progress_utils.py show --cache-dir /path/to/cache

# List all cache directories with progress files
python progress_utils.py list

# Reset failed trajectories to retry them
python progress_utils.py reset-failed --cache-dir /path/to/cache

# Reset all progress (start fresh)
python progress_utils.py reset --cache-dir /path/to/cache
```

## Progress File Structure

The progress is stored in `progress.json` within each cache directory:

```json
{
  "dataset_name": "example_dataset",
  "start_time": 1641234567.89,
  "last_update": 1641234890.12,
  "completed_trajectories": ["traj_001", "traj_002"],
  "failed_trajectories": ["traj_003"],
  "total_chunks": 1250,
  "processing_config": {
    "max_chunk_distance_m": 50.0,
    "overlap_distance_m": 1.0,
    "min_chunk_distance_m": 0.3,
    "dino_model_type": "large"
  },
  "trajectory_checksums": {
    "traj_001": "a1b2c3d4e5f6g7h8",
    "traj_002": "h8g7f6e5d4c3b2a1"
  }
}
```

## Resuming Interrupted Processing

When processing is interrupted:

1. **Restart the script** with the same configuration
2. **Automatic detection**: The system detects existing progress
3. **Validation**: Checks if configuration and data haven't changed
4. **Resume**: Continues from the last completed trajectory

Example output when resuming:
```
Loaded existing progress: 150 trajectories completed
Resuming processing: 150/500 trajectories completed
Remaining trajectories: 350
```

## Handling Configuration Changes

If you need to change processing parameters:

1. **Force rebuild**: Use `force_rebuild=True` in the config or code
2. **Reset progress**: Use `python progress_utils.py reset --cache-dir /path/to/cache`
3. **Manual confirmation**: The system will prompt you to confirm continuing with changed config

## Error Recovery

### Failed Trajectories

When trajectories fail:
- They are marked as failed but not removed from the processing queue
- Detailed error messages are logged
- Failed trajectories can be retried by resetting their status

### Retry Failed Trajectories

```bash
# Reset only failed trajectories
python progress_utils.py reset-failed --cache-dir /path/to/cache

# Then restart processing
python dino_cache.py --config config/data/dino_cache.yaml
```

## Performance Benefits

### Memory Efficiency
- Processes trajectories one at a time
- Releases memory after each trajectory
- Prevents memory accumulation over long runs

### Time Savings
- No need to restart from the beginning after interruptions
- Skip already processed trajectories
- Efficient progress tracking with minimal overhead

### Reliability
- Atomic file operations prevent corruption
- Checksum validation ensures data integrity
- Graceful handling of individual failures

## Monitoring Progress

### Real-time Updates

The system provides periodic progress updates:
```
Progress: 150/500 trajectories (30.0%), 1250 chunks, 2.5h elapsed
```

### Detailed Progress View

```bash
python progress_utils.py show --cache-dir /path/to/cache
```

Output:
```
📊 Progress Status for /path/to/cache
============================================================
Dataset: example_dataset
Completed trajectories: 150
Failed trajectories: 5
Total chunks generated: 1250
Processing time: 2.5 hours
Last update: Mon Jan 10 15:30:45 2022

Processing Configuration:
  max_chunk_distance_m: 50.0
  overlap_distance_m: 1.0
  min_chunk_distance_m: 0.3
  dino_model_type: large

❌ Failed Trajectories (5):
  - trajectory_with_corrupted_data
  - trajectory_with_insufficient_movement
  ...
```

## Best Practices

1. **Regular Monitoring**: Check progress periodically using `progress_utils.py show`
2. **Handle Failures**: Investigate and retry failed trajectories
3. **Backup Progress**: The `progress.json` file is small and can be backed up
4. **Clean Restarts**: Use `reset` command when changing critical parameters
5. **Resource Management**: Monitor disk space and memory usage during long runs

## Troubleshooting

### Common Issues

1. **Permission Errors**: Ensure write access to cache directory
2. **Disk Space**: Monitor available disk space for progress files
3. **Configuration Conflicts**: Reset progress when changing critical parameters
4. **Memory Issues**: Individual trajectory failures won't affect overall progress

### Debug Mode

For debugging, you can examine the progress file directly:
```bash
cat /path/to/cache/progress.json | python -m json.tool
```

This progress tracking system makes DINO cache preprocessing robust, reliable, and resumable, especially important for large-scale dataset processing that may take hours or days to complete.
