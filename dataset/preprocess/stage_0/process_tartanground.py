import os
import json
import shutil
import math
from glob import glob
from tqdm import tqdm


def extract_yaw_from_quaternion(qx, qy, qz, qw):
    """Convert quaternion to yaw (heading in radians)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def convert_dataset(input_base, output_base):
    scenes = [d for d in os.listdir(input_base) if os.path.isdir(os.path.join(input_base, d))]

    for scene in scenes:
        scene_path = os.path.join(input_base, scene, "Data_ground")
        if not os.path.exists(scene_path):
            continue

        trajectories = [d for d in os.listdir(scene_path) if d.startswith("P")]

        for traj in tqdm(trajectories, desc=f"Processing {scene}"):
            traj_path = os.path.join(scene_path, traj)
            image_dir = os.path.join(traj_path, "image_lcam_front")
            pose_file = os.path.join(traj_path, "pose_lcam_front.txt")

            if not os.path.exists(image_dir) or not os.path.exists(pose_file):
                continue

            new_traj_name = f"{scene}_{traj}"
            new_traj_path = os.path.join(output_base, new_traj_name)
            os.makedirs(new_traj_path, exist_ok=True)

            # Move images
            image_files = sorted(glob(os.path.join(image_dir, "*.png")))
            for img in image_files:
                shutil.move(img, os.path.join(new_traj_path, os.path.basename(img)))

            # Process poses
            positions = []
            yaws = []

            with open(pose_file, "r") as f:
                for line in f:
                    elems = list(map(float, line.strip().split()))
                    if len(elems) != 7:
                        continue
                    x, y = elems[0], elems[1]
                    qx, qy, qz, qw = elems[3], elems[4], elems[5], elems[6]
                    yaw = extract_yaw_from_quaternion(qx, qy, qz, qw)

                    positions.append([x, y])
                    yaws.append(yaw)

            # Save JSON
            traj_data = {
                "position": positions,
                "yaw": yaws
            }

            with open(os.path.join(new_traj_path, "traj_data.json"), "w") as f:
                json.dump(traj_data, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert and move dataset format.")
    parser.add_argument("input_base", help="Path to the original dataset base folder")
    parser.add_argument("output_base", help="Path to the new dataset base folder")
    args = parser.parse_args()

    convert_dataset(args.input_base, args.output_base)
