#!/usr/bin/env python3
import os
import json
import random
import torch
import argparse

def load_traj_data(traj_dir):
    """
    Load `traj_data.json` in the trajectory directory.
    Returns a dict with keys "position" (list of [x,y]) and "yaw" (list of float).
    """
    path = os.path.join(traj_dir, "traj_data.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find traj_data.json in {traj_dir}")
    with open(path, "r") as f:
        return json.load(f)


def find_middle_chunk_files(cache_folder):
    """
    Return a list of all .pt files under cache_folder whose chunk‐ID suffix is NOT "00000"
    (i.e. skip the very first chunk of each trajectory).
    """
    all_pts = []
    for fname in os.listdir(cache_folder):
        if not fname.endswith(".pt"):
            continue
        # if the filename ends with "_00000.pt", skip it
        if fname.endswith("_00000.pt"):
            continue
        all_pts.append(os.path.join(cache_folder, fname))

    if not all_pts:
        raise RuntimeError(f"No non‐zero‐start .pt files found in {cache_folder}")
    return all_pts


def inspect_random_middle_chunk(data_root, cache_folder, n_samples=1):
    """
    Pick `n_samples` random .pt files from cache_folder whose start‐index ≠ 0.
    For each chosen chunk:
      - Load the dictionary inside (.pt → {"features": Tensor(L,D),
                                           "positions": [ {“position”:[x,y], “yaw”:…}, … ],
                                           "indices":   [ i_0, i_1, … ]}
    )
      - Print: feature count (L), feature dimension (D), and for each local i:
             * global_idx = indices[i]
             * saved position = positions[i]["position"]
             * saved yaw      = positions[i]["yaw"]  (if present)
             * ground‐truth (x,y,yaw) from data_root/<traj_name>/traj_data.json
    """

    # 1) list only the chunks that do NOT start at index 0
    middle_chunks = find_middle_chunk_files(cache_folder)
    picks = random.sample(middle_chunks, min(n_samples, len(middle_chunks)))

    for pt_path in picks:
        print(f"\n=== Inspecting middle‐chunk file: {pt_path} ===")

        # chunk_id is "<traj_name>_<start_idx>"
        chunk_fname = os.path.basename(pt_path)
        if not chunk_fname.endswith(".pt"):
            raise RuntimeError(f"Unexpected file: {chunk_fname}")
        chunk_id = chunk_fname[:-3]  # strip ".pt"
        # everything before the final "_" is the trajectory name
        traj_name = chunk_id.rsplit("_", 1)[0]
        print(f"  • chunk_id = {chunk_id}")
        print(f"  • inferred trajectory name = {traj_name}")

        traj_dir = os.path.join(data_root, traj_name)
        if not os.path.isdir(traj_dir):
            raise FileNotFoundError(f"Missing trajectory directory: {traj_dir}")

        # load ground‐truth traj_data.json
        traj_data = load_traj_data(traj_dir)
        gt_positions = traj_data.get("position", [])
        gt_yaws      = traj_data.get("yaw", [])

        # load the .pt file
        data = torch.load(pt_path, map_location="cpu")
        assert "features"  in data and "positions" in data and "indices" in data, \
            f"chunk {chunk_id} missing required keys"

        feats = data["features"]         # shape (L, D)
        pos_list = data["positions"]     # should be list of dicts
        idx_list = data["indices"]       # list of ints

        if not isinstance(feats, torch.Tensor):
            raise TypeError(f"'features' is {type(feats)}, expected torch.Tensor")

        L, D = feats.shape
        print(f"  → features.shape = {feats.shape}  (L={L}, feature_dim={D})")
        print(f"  → saved positions: len(positions) = {len(pos_list)}")
        print(f"  → saved indices:   len(indices)   = {len(idx_list)}")

        if len(pos_list) != L or len(idx_list) != L:
            raise ValueError(
                f"Lengths mismatch in {chunk_id}: "
                f"features has {L} rows but positions has {len(pos_list)} and indices has {len(idx_list)}"
            )

        # For each local entry, compare the saved (position, yaw) with ground-truth
        print("\n    [LOCAL_IDX] → global_idx | saved_position, saved_yaw ||  gt_position, gt_yaw")
        print("    ---------------------------------------------------------------")
        for local_i in range(L):
            global_idx  = idx_list[local_i]
            saved_entry = pos_list[local_i]

            # saved_entry can be either a dict with keys {"position":[x,y], "yaw":val}
            # or just a list [x,y], depending on how your code wrote it.
            if isinstance(saved_entry, dict) and "position" in saved_entry:
                saved_xy  = saved_entry["position"]
                saved_yaw = saved_entry.get("yaw", None)
            else:
                # fallback if positions were saved as `[x, y]` with no yaw
                saved_xy  = saved_entry
                saved_yaw = None

            if not (0 <= global_idx < len(gt_positions)):
                print(f"    ! [WARNING] local {local_i}: global_idx {global_idx} is out of range (N={len(gt_positions)})")
                gt_xy, gt_yaw = None, None
            else:
                gt_xy  = gt_positions[global_idx]
                gt_yaw = gt_yaws[global_idx] if global_idx < len(gt_yaws) else None

            print(f"    [{local_i:3d}] → {global_idx:4d} | "
                  f"saved_xy={saved_xy}, saved_yaw={saved_yaw}  ||  "
                  f"gt_xy={gt_xy}, gt_yaw={gt_yaw}")
        print("    ---------------------------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect random “middle” .pt files (i.e. start‐index ≠ 0) in a DINO cache."
    )
    parser.add_argument(
        "--data-root",   required=True,
        help="Root data folder where each trajectory folder lives (e.g. /app/Sati_data/Recon_320x240)."
    )
    parser.add_argument(
        "--cache-folder", required=True,
        help="The DINO‐cache folder (where cache_metadata.json and all *.pt files live)."
    )
    parser.add_argument(
        "--n-samples",   default=1, type=int,
        help="How many random “middle” .pt files to inspect (default: 1)."
    )
    args = parser.parse_args()

    inspect_random_middle_chunk(args.data_root, args.cache_folder, n_samples=args.n_samples)