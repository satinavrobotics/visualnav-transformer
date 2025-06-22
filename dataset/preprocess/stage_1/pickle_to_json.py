#!/usr/bin/env python3
import os
import pickle
import json
import argparse
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def convert_pkl_to_json(pkl_path, json_path):
    """
    Loads a pickle file and writes its content to a JSON file.
    Assumes the pickle contains a dictionary.
    """
    try:
        with open(pkl_path, 'rb') as pf:
            data = pickle.load(pf)
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        return

    if not isinstance(data, dict):
        print(f"Warning: {pkl_path} does not contain a dictionary. Skipping.")
        return

    try:
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(data, jf, indent=2, cls=NumpyEncoder)
        print(f"Converted {pkl_path} -> {json_path}")
    except Exception as e:
        print(f"Error writing {json_path}: {e}")

def process_trajectory_folder(folder):
    """
    If the folder contains a traj_data.pkl file, convert it to JSON.
    """
    pkl_file = os.path.join(folder, 'traj_data.pkl')
    if os.path.isfile(pkl_file):
        json_file = os.path.join(folder, 'traj_data.json')
        convert_pkl_to_json(pkl_file, json_file)

def process_directory(root_dir):
    """
    Recursively walk through root_dir and process every folder that contains a traj_data.pkl.
    """
    for current_root, dirs, files in os.walk(root_dir):
        if 'traj_data.pkl' in files:
            process_trajectory_folder(current_root)

def main():
    parser = argparse.ArgumentParser(
        description="Convert all traj_data.pkl files in a given folder (recursively) to JSON."
    )
    parser.add_argument('--i', required=True, help="Path to the input folder containing trajectory folders.")
    args = parser.parse_args()
    process_directory(args.i)

if __name__ == '__main__':
    main()