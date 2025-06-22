import os
import sys
import glob
import json


"""
Checks if a given trajectory folder has the correct number of images and pose json entries.
"""

def find_rgb_traj_folders(root_dir):
    """
    Find rgb_* folders that contain traj_data.json.
    Return list of (rgb_folder, dataset_root) tuples.
    """
    traj_folders = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'traj_data.json' in filenames and os.path.basename(dirpath).startswith('rgb'):
            dataset_parts = dirpath.split(os.sep)
            dataset_root = None
            for i in reversed(range(len(dataset_parts))):
                candidate = os.path.join(*dataset_parts[:i+1])
                if os.path.exists(os.path.join(candidate, "dataset_metadata.json")):
                    dataset_root = candidate
                    break
            if dataset_root is None:
                dataset_root = root_dir  # fallback if not found
            traj_folders.append((dirpath, dataset_root))
    return traj_folders

def check_image_vs_json_count(traj_folder):
    """
    Return (traj_folder, num_images, num_positions, difference)
    """
    image_files = glob.glob(os.path.join(traj_folder, "*.jpg"))
    num_images = len(image_files)

    traj_json_path = os.path.join(traj_folder, "traj_data.json")
    if not os.path.exists(traj_json_path):
        print(f"[ERROR] traj_data.json not found in {traj_folder}")
        return (traj_folder, num_images, None, None)

    with open(traj_json_path, "r") as f:
        data = json.load(f)
        positions = data.get("position", [])
        num_positions = len(positions)

    diff = abs(num_images - num_positions)
    return (traj_folder, num_images, num_positions, diff)

def check_all_trajectory_folders(root_dir):
    """
    Check all folders, print detailed mismatches, and summarize.
    """
    traj_folders = find_rgb_traj_folders(root_dir)
    if not traj_folders:
        print("No trajectory folders found.")
        return

    total = 0
    exact = 0
    diff_2 = 0
    diff_gt_2 = 0

    for rgb_folder, _ in traj_folders:
        total += 1
        folder, imgs, poses, diff = check_image_vs_json_count(rgb_folder)

        if poses is None:
            continue

        if diff == 0:
            exact += 1
        elif diff == 2:
            diff_2 += 1
            print(f"[DIFF 2] {folder} — Images: {imgs}, Positions: {poses}")
        elif diff > 2:
            diff_gt_2 += 1
            print(f"[DIFF >2] {folder} — Images: {imgs}, Positions: {poses}")

    print("\n📊 Summary")
    print(f"  Total folders checked    : {total}")
    print(f"  ✅ Exact matches          : {exact}")
    print(f"  ⚠️  Diff = 2              : {diff_2}")
    print(f"  ❗ Diff > 2               : {diff_gt_2}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_image_vs_json_diff.py <dataset_root_directory>")
        sys.exit(1)

    root_directory = sys.argv[1]
    check_all_trajectory_folders(root_directory)