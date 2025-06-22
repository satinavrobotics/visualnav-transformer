#!/usr/bin/env python3
"""
Used for certain datasets, where the trajectory data is nested.
flatten_single_subdir.py

In each immediate subfolder of DATASET_ROOT, if there is exactly one subdirectory
(e.g. "node_*", "unknown_*", etc.), move all files & folders from that single
subdirectory up one level, then remove the now-empty subdirectory.
"""

import os
import shutil
import argparse

def flatten_single_subdir(dataset_root: str):
    for entry in sorted(os.listdir(dataset_root)):
        chunk_dir = os.path.join(dataset_root, entry)
        if not os.path.isdir(chunk_dir):
            continue

        # list only *directories* inside this chunk
        children = [
            name for name in os.listdir(chunk_dir)
            if os.path.isdir(os.path.join(chunk_dir, name))
        ]

        # only flatten if exactly one child-dir
        if len(children) != 1:
            continue

        child = children[0]
        child_dir = os.path.join(chunk_dir, child)

        # move everything from child_dir → chunk_dir
        for item in os.listdir(child_dir):
            src = os.path.join(child_dir, item)
            dst = os.path.join(chunk_dir, item)
            # avoid overwriting
            if os.path.exists(dst):
                dst = os.path.join(chunk_dir, f"dup_{item}")
            shutil.move(src, dst)

        # delete the empty child folder
        os.rmdir(child_dir)
        print(f"Flattened '{child}' → '{entry}'")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Flatten any single subdirectory in each chunk into its parent."
    )
    p.add_argument(
        "dataset_root",
        help="Path to the dataset root (e.g. GND_320x240) containing chunk_* folders"
    )
    args = p.parse_args()
    flatten_single_subdir(args.dataset_root)
    print("All done.")