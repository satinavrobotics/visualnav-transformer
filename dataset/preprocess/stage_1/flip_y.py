import os
import json

def flip_y_axis_in_traj(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Flip the y-axis (negate the second value in each [x, y] pair)
    flipped_positions = [[x, -y] for x, y in data.get('position', [])]
    data['position'] = flipped_positions

    # Save the updated data back to the file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def process_directory(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == 'traj_data.json':
                file_path = os.path.join(dirpath, filename)
                print(f"Processing: {file_path}")
                flip_y_axis_in_traj(file_path)

# Example usage:
# Replace this with your actual root directory
root_directory = "/mnt/sati-data/Sati_data_320x240/TartanGround/"
process_directory(root_directory)
