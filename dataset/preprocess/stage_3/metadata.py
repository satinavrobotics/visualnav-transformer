import os
import json
import argparse

def is_image_file(filename):
    """Check if the file is an image by extension."""
    return filename.lower().endswith((".png", ".jpg", ".jpeg"))

def count_images_in_folder(folder_path):
    """Count how many image files (.png, .jpg, .jpeg) are in folder_path (non-recursive)."""
    return sum(1 for f in os.listdir(folder_path) if is_image_file(f))

def find_trajectory_folders(root_dir):
    """Recursively find directories that contain traj_data.json."""
    traj_folders = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'traj_data.json' in filenames:
            traj_folders.append(dirpath)
    return traj_folders

def extract_dataset_name(folder_name):
    """Extract dataset name from the folder name (before the first underscore)."""
    return folder_name.split('_')[0]

def generate_metadata(root_dir, dataset_name):
    """Generate metadata with dataset root and trajectory details."""
    trajectory_info = []
    total_images = 0
    traj_folders = find_trajectory_folders(root_dir)
    
    for traj_folder in traj_folders:
        image_count = count_images_in_folder(traj_folder)
        total_images += image_count
        trajectory_info.append({
            "path": os.path.relpath(traj_folder, start=root_dir),  # Store relative path to root_dir
            "cnt": image_count
        })
    
    # Assuming 30 images per second, calculate minutes from total images
    seconds_per_image = 1 / 10
    total_minutes = (total_images * seconds_per_image) / 60
    
    return {
        "name": dataset_name,
        "root": root_dir,
        "traj_cnt": len(traj_folders),
        "img_cnt": total_images,
        "duration": total_minutes,
        "trajectories": trajectory_info
    }

def main(root_dir, output_file):
    """Main function to create metadata JSON."""
    # Extract dataset name from root_dir's name
    dataset_name = extract_dataset_name(os.path.basename(root_dir))
    
    metadata = generate_metadata(root_dir, dataset_name)
    
    output_path = os.path.join(root_dir, output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata for dataset.")
    parser.add_argument("--i", required=True, help="Path to dataset folder.")
    parser.add_argument("--o", default="dataset_metadata.json", help="Output metadata JSON file.")
    args = parser.parse_args()
    
    main(args.i, args.o)
