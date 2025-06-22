import json
import numpy as np

# 1) load the positions
traj = json.load(open("/app/Sati_data/SIT_320x240/rgb_images_Cafeteria_3/traj_data.json"))
positions = np.array(traj["position"])   # shape (N,2)

# 2) compute pairwise displacements
diffs = positions[1:] - positions[:-1]   # shape (N-1,2)
dists = np.linalg.norm(diffs, axis=1)    # shape (N-1,)

# 3) report stats
print(f"mean spacing   = {dists.mean():.3f} m")
print(f"median spacing = {np.median(dists):.3f} m")
