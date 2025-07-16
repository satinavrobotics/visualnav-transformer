import os, json
import numpy as np

# ─── CONFIG ────────────────────────────────────────────────────────────────
# Path to the same data_config.yaml you already use:
DATA_CONFIG = "/app/visualnav-transformer/config/data/data_config.yaml"
# Typical robot speed (m/s) you want to assume for fps estimation:
ASSUMED_SPEED = 0.5
# Where to write the output
OUTPUT_PATH = "/app/Sati_data/fps_estimates.json"
# ────────────────────────────────────────────────────────────────────────────

import yaml

with open(DATA_CONFIG, "r") as f:
    cfg = yaml.safe_load(f)["datasets"]

results = {}
for name, d in cfg.items():
    if not d.get("available", False):
        continue

    root = d["data_folder"]
    mdists = []
    myaws  = []

    # walk every trajectory under root (using your dataset_metadata.json)
    meta = os.path.join(root, "dataset_metadata.json")
    if not os.path.exists(meta): 
        continue
    with open(meta,"r") as f:
        trajs = json.load(f)["trajectories"]

    for t in trajs:
        traj_dir = os.path.join(root, t["path"])
        jd = os.path.join(traj_dir, "traj_data.json")
        if not os.path.exists(jd):
            continue
        td = json.load(open(jd,"r"))
        pos = np.array(td["position"], dtype=float)   # (N,3) or (N,2)
        yaw = np.array(td.get("yaw", [0]*len(pos)), dtype=float)

        if len(pos)<2: 
            continue

        # compute displacements & yaw diffs
        # raw per‐frame displacement & yaw change
        ds = np.linalg.norm(pos[1:,:2] - pos[:-1,:2], axis=1)
        ys = np.abs(np.diff(yaw))

        # filter out zero (or near‐zero) repeats
        eps = 1e-30
        ds_nz = ds[ds > eps]
        ys_nz = ys[ys > eps]
        if len(ds_nz) == 0:
            # nothing but repeats
            continue
    

        mdists.append(np.median(ds_nz))
        myaws .append(np.median(ys_nz) if len(ys_nz) else 0.0)

    if not mdists:
        continue

    median_disp = float(np.median(mdists))
    median_yaw  = float(np.median(myaws))
    est_fps     = ASSUMED_SPEED / median_disp if median_disp>0 else None
    
    frames_per_30cm = 0.3 / median_disp if median_disp > 0 else None

    results[name] = {
        "median_disp_m":  median_disp,
        "median_yaw_rad": median_yaw,
        "assumed_speed_mps": ASSUMED_SPEED,
        "estimated_fps": est_fps,
        "frames_per_0.3m":    frames_per_30cm,
        "num_trajectories": len(mdists)
    }

# write out
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Wrote fps estimates for {len(results)} datasets to {OUTPUT_PATH}")