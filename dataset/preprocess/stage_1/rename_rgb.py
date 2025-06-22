#!/usr/bin/env python3
import os
import argparse
import shutil
import re

def strip_image_prefixes(root_dir):
    """
    Strip filename prefixes like 'rgb_320x240_0.jpg' to '0.jpg'
    for images inside rgb_* folders found via find_rgb_traj_folders().
    """
    rgb_folders = [f[0] for f in find_rgb_traj_folders(root_dir)]
    image_exts = {'.jpg', '.jpeg', '.png'}

    total_renamed = 0

    for folder in rgb_folders:
        for fname in os.listdir(folder):
            name, ext = os.path.splitext(fname)
            if ext.lower() in image_exts:
                # Match trailing digits only (e.g., rgb_320x240_12 → 12)
                match = re.search(r'(\d+)$', name)
                if match:
                    new_name = match.group(1) + ext.lower()
                    src_path = os.path.join(folder, fname)
                    dst_path = os.path.join(folder, new_name)
                    if os.path.exists(dst_path):
                        print(f"[SKIP] Destination exists: {dst_path}")
                        continue
                    os.rename(src_path, dst_path)
                    print(f"[RENAME] {fname} → {new_name}")
                    total_renamed += 1
                else:
                    print(f"[WARN] Could not parse index from: {fname}")

    print(f"\n✅ Renamed {total_renamed} image(s) in {len(rgb_folders)} rgb_* folders.")

def find_rgb_traj_folders(root_dir):
    """
    Find rgb_* folders that contain traj_data.json.
    Return list of (rgb_folder, dataset_root) tuples.
    """
    traj_folders = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'traj_data.json' in filenames and os.path.basename(dirpath).startswith('rgb'):
            dataset_root = dirpath.split(os.sep)
            for i in reversed(range(len(dataset_root))):
                if os.path.exists(os.path.join(*dataset_root[:i+1], "dataset_metadata.json")):
                    dataset_root = os.path.join(*dataset_root[:i+1])
                    break
            traj_folders.append((dirpath, dataset_root))
    return traj_folders


def rename_folder_with_prefix(folder_path, prefix="rgb_"):
    """
    Renames the folder by adding a prefix to its basename.
    For example, if folder_path is "/path/to/traj_folder", and its basename
    does not already start with prefix, it is renamed to "/path/to/rgb_traj_folder".
    Returns the new folder path (or original if no change is needed).
    """
    parent_dir = os.path.dirname(folder_path)
    base_name = os.path.basename(folder_path)
    if base_name.startswith(prefix):
        # Already renamed
        return folder_path
    new_base = prefix + base_name
    new_folder_path = os.path.join(parent_dir, new_base)
    os.rename(folder_path, new_folder_path)
    print(f"Renamed: {folder_path} -> {new_folder_path}")
    return new_folder_path

def rename():
    parser = argparse.ArgumentParser(
        description="Recursively rename trajectory folders containing JPEG images by adding an 'rgb_' prefix."
    )
    parser.add_argument("--input-dir", "-i", required=True,
                        help="Input folder to search for trajectory folders")
    args = parser.parse_args()

    # Find all folders that contain at least one JPEG file
    traj_folders = find_rgb_traj_folders(args.input_dir)
    print(f"Found {len(traj_folders)} trajectory folders containing JPEG images.")

    # To safely rename folders, process from deepest to shallowest.
    traj_folders.sort(key=lambda p: len(p), reverse=True)

    for folder in traj_folders:
        rename_folder_with_prefix(folder, prefix="rgb_")
        
        
def move_traj(root_dir):
    """
    Locates the 'traj_data.json' in trajectory folders and moves it into
    a subfolder starting with 'rgb_'.
    """
    # Find all trajectory folders
    traj_folders = find_rgb_traj_folders(root_dir)
    print(f"Found {len(traj_folders)} trajectory folders containing JPEG images.")

    # Process each trajectory folder
    for traj_folder in traj_folders:
        traj_data_path = os.path.join(traj_folder, 'traj_data.json')
        
        # Check if traj_data.json exists in the folder
        if os.path.exists(traj_data_path):
            # Find the subfolder starting with 'rgb_'
            rgb_folder = None
            for subdir in os.listdir(traj_folder):
                if subdir.startswith('rgb_') and os.path.isdir(os.path.join(traj_folder, subdir)):
                    rgb_folder = os.path.join(traj_folder, subdir)
                    break

            # Move traj_data.json into the rgb_ folder
            new_traj_data_path = os.path.join(rgb_folder, 'traj_data.json')
            shutil.move(traj_data_path, new_traj_data_path)
            print(f"Moved {traj_data_path} to {new_traj_data_path}")
        else:
            print(f"Warning: traj_data.json not found in {traj_folder}")

def copy_traj_and_rgb(src_root, dest_root):
    traj_folders = find_rgb_traj_folders(src_root)
    print(f"[INFO] Found {len(traj_folders)} rgb_* folders with traj_data.json")

    copied_datasets = set()

    for src_rgb_folder, dataset_root in traj_folders:
        # Copy dataset_metadata.json (once per dataset)
        if dataset_root not in copied_datasets:
            meta_src = os.path.join(dataset_root, 'dataset_metadata.json')
            rel_dataset = os.path.relpath(dataset_root, src_root)
            meta_dst_dir = os.path.join(dest_root, rel_dataset)
            os.makedirs(meta_dst_dir, exist_ok=True)
            if os.path.exists(meta_src):
                shutil.copy(meta_src, os.path.join(meta_dst_dir, 'dataset_metadata.json'))
                print(f"[META] Copied: {meta_src} -> {meta_dst_dir}")
            copied_datasets.add(dataset_root)

        # Copy the full rgb_ folder
        rel_path = os.path.relpath(src_rgb_folder, src_root)
        dst_rgb_folder = os.path.join(dest_root, rel_path)
        os.makedirs(os.path.dirname(dst_rgb_folder), exist_ok=True)
        shutil.copytree(src_rgb_folder, dst_rgb_folder, dirs_exist_ok=True)
        print(f"[RGB] Copied folder: {src_rgb_folder} -> {dst_rgb_folder}")

 
if __name__ == "__main__":
    # rename()
    dir = "Sati_data/MuSoHu_320x240"
    # move_traj(dir)
    # copy_traj_and_rgb("Sati_data/Sati_data_320x240", "Sati_data/Sati_data_rgb")
    # copy_traj_and_rgb("app/Sati_data_320x240", "/mnt/sati-data")
    # copy_traj_and_rgb("app/Sati_data_og", "/mnt/sati-data")
    parser = argparse.ArgumentParser(
        description="Prepend 'rgb_' to every subfolder in a given directory"
    )
    parser.add_argument("-i", "--input-dir", required=True,
                        help="Root directory whose child folders you want to rename")
    parser.add_argument("-p", "--prefix", default="rgb_",
                        help="Prefix to add")
    args = parser.parse_args()

    # Rename each immediate subfolder under args.input_dir
    for name in os.listdir(args.input_dir):
        full = os.path.join(args.input_dir, name)
        if os.path.isdir(full):
            rename_folder_with_prefix(full, prefix=args.prefix)
    
    