import os
import cv2
import json
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import natsort


def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    

def to_local_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos).dot(rotmat)


def render_plot(position_list, current_index, figsize=(5, 5)):
    fig = Figure(figsize=figsize)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)
    xs, ys = zip(*position_list)
    
    ax.plot(xs, ys, color="gray", linewidth=1, label="Trajectory")
    ax.plot(xs[current_index], ys[current_index], 'ro', markersize=6, label="Current Position")
    
    ax.set_aspect("equal")
    
    padding = 1
    ax.set_xlim(min(xs) - padding, max(xs) + padding)
    ax.set_ylim(min(ys) - padding, max(ys) + padding)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    
    fig.tight_layout(pad=0.5)
    canvas.draw()

    # Get the RGBA buffer and convert to RGB
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)
    # Convert from RGBA to RGB by dropping the alpha channel
    img = img[:, :, :3]
    return img


def visualize_trajectory(traj_folder, output_video):
    traj_json_path = os.path.join(traj_folder, "traj_data.json")
    if not os.path.exists(traj_json_path):
        raise FileNotFoundError(f"{traj_json_path} not found")

    with open(traj_json_path, "r") as f:
        traj_data = json.load(f)

    positions = traj_data["position"]
    yaws = traj_data["yaw"]
    print(f"Found {len(positions)} positions in {traj_json_path}")
    image_files = natsort.natsorted([f for f in os.listdir(traj_folder) if f.endswith(".jpg")])
    print(f"Found {len(image_files)} images in {traj_folder}")

    if len(positions) != len(image_files):
        raise ValueError("Mismatch between number of images and trajectory points.")

    start = 10
    end = min(len(positions), start+200)
    image_paths = [os.path.join(traj_folder, f) for f in image_files][start:end]
    positions = to_local_coords(np.array(positions[start:end]), positions[start], yaws[start])

    sample_img = cv2.imread(image_paths[0])
    h, w, _ = sample_img.shape

    plot_img = render_plot(positions, 0)
    plot_h, plot_w, _ = plot_img.shape

    combined_width = w + plot_w
    combined_height = max(h, plot_h)

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 15, (combined_width, combined_height))

    for i in tqdm(range(len(image_paths)), desc="Generating video"):
        img = cv2.imread(image_paths[i])
        if img is None:
            print(f"Failed to load image {image_paths[i]}")
            continue

        plot_img = render_plot(positions, i)
        plot_img = cv2.resize(plot_img, (plot_w, h))

        combined = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        combined[:h, :w] = img
        combined[:h, w:] = plot_img

        out.write(combined)

    out.release()
    print(f"Video saved to: {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trajectory folder into video.")
    parser.add_argument("traj_folder", help="Path to trajectory folder (containing .png and traj_data.json)")
    parser.add_argument("output_video", help="Path to save the output .mp4 video")
    args = parser.parse_args()

    visualize_trajectory(args.traj_folder, args.output_video)
