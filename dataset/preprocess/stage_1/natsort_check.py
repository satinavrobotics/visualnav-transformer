import os
import sys
import json
from natsort import natsorted

def rename_files_sequentially(folder, prefix="frame_", ext=None, padding=6):
    files = os.listdir(folder)
    # If ext is given, filter by that extension
    if ext:
        files = [f for f in files if f.lower().endswith(ext.lower())]

    sorted_files = natsorted(files)

    for i, fname in enumerate(sorted_files, start=1):
        # Use original extension if ext is None
        file_ext = ext if ext else os.path.splitext(fname)[1]
        new_fname = f"{prefix}{str(i).zfill(padding)}{file_ext}"

        old_path = os.path.join(folder, fname)
        new_path = os.path.join(folder, new_fname)

        if old_path != new_path:
            # Avoid overwriting existing files (could happen if naming conflicts)
            if os.path.exists(new_path):
                print(f"Skipping rename to {new_fname} because it already exists.")
                continue
            print(f"Renaming {fname} -> {new_fname}")
            os.rename(old_path, new_path)

def check_and_fix_trajectory_order(base_dir):
    metadata_path = os.path.join(base_dir, "dataset_metadata.json")
    if not os.path.isfile(metadata_path):
        print(f"Error: dataset_metadata.json not found in {base_dir}")
        return

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    trajectories = metadata.get("trajectories", [])
    total = len(trajectories)
    rename_needed = 0

    for traj in trajectories:
        traj_path = os.path.join(base_dir, traj["path"])
        if not os.path.isdir(traj_path):
            print(f"Warning: Trajectory path does not exist: {traj_path}")
            continue

        files = os.listdir(traj_path)
        sorted_files = sorted(files)
        natsorted_files = natsorted(files)

        if sorted_files != natsorted_files:
            rename_needed += 1
            print(f"Trajectory '{traj['path']}' ordering mismatch. Renaming files sequentially...")
            # Optionally, detect extension from first file or leave as None to keep original extensions
            example_ext = os.path.splitext(natsorted_files[0])[1] if natsorted_files else None
            rename_files_sequentially(traj_path, ext=example_ext)

    print(f"Trajectories needing renaming: {rename_needed}/{total}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_trajectory_order.py <directory>")
        sys.exit(1)

    base_directory = sys.argv[1]
    check_and_fix_trajectory_order(base_directory)
