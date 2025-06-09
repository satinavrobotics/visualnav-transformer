#!/usr/bin/env python3

import os
import torch
import numpy as np
import argparse

def verify_pt_file(pt_file_path):
    """
    Verify the structure of a .pt file from the DINO cache.
    
    Args:
        pt_file_path: Path to the .pt file
    """
    print(f"Verifying .pt file: {pt_file_path}")
    
    # Load the .pt file
    try:
        data = torch.load(pt_file_path, weights_only=False)
        print(f"Successfully loaded .pt file")
    except Exception as e:
        print(f"Error loading .pt file: {e}")
        return
    
    # Check the structure
    print("\nFile structure:")
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor of shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"  {key}: {type(value)}")
                
        # If there are features and positions, print some samples
        if "features" in data and isinstance(data["features"], torch.Tensor):
            features = data["features"]
            print(f"\nFeature samples (first 3 rows, first 5 values):")
            for i in range(min(3, features.shape[0])):
                print(f"  Row {i}: {features[i, 0:5].tolist()}")
                
        if "positions" in data and isinstance(data["positions"], list):
            positions = data["positions"]
            print(f"\nPosition samples (first 3):")
            for i in range(min(3, len(positions))):
                print(f"  Position {i}: {positions[i]}")
    else:
        print(f"  Data is not a dictionary, but a {type(data)}")
        
    print("\nVerification complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify DINO Feature Cache .pt File")
    
    parser.add_argument(
        "--pt_file",
        type=str,
        required=True,
        help="Path to the .pt file to verify"
    )
    
    args = parser.parse_args()
    
    verify_pt_file(args.pt_file)
