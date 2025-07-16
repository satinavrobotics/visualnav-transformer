import os
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import yaml

from visualnav_transformer.train.vint_train.visualizing.visualize_utils import (
    CYAN,
    GREEN,
    MAGENTA,
    RED,
    VIZ_IMAGE_SIZE,
    numpy_to_img,
)

from visualnav_transformer import ROOT_TRAIN
with open(
        # CHANGE
    # os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r"
    #os.path.join(ROOT_TRAIN, "vint_train/data/data_config.yaml"), "r"
    os.path.join("/app/visualnav-transformer/config/data/data_config.yaml"), "r"
) as f:
    data_config = yaml.safe_load(f)["datasets"]


def visualize_traj_pred(
    batch_obs_images: np.ndarray,
    batch_goal_images: np.ndarray,
    dataset_indices: np.ndarray,
    batch_goals: np.ndarray,
    batch_pred_waypoints: np.ndarray,
    batch_label_waypoints: np.ndarray,
    eval_type: str,
    normalized: bool,
    save_folder: str,
    epoch: int,
    num_images_preds: int = 8,
    use_mlflow: bool = True,
    display: bool = False,
):
    """
    Compare predicted path with the gt path of waypoints using egocentric visualization. This visualization is for the last batch in the dataset.

    Args:
        batch_obs_images (np.ndarray): batch of observation images [batch_size, height, width, channels]
        batch_goal_images (np.ndarray): batch of goal images [batch_size, height, width, channels]
        dataset_names: indices corresponding to the dataset name
        batch_goals (np.ndarray): batch of goal positions [batch_size, 2]
        batch_pred_waypoints (np.ndarray): batch of predicted waypoints [batch_size, horizon, 4] or [batch_size, horizon, 2] or [batch_size, num_trajs_sampled horizon, {2 or 4}]
        batch_label_waypoints (np.ndarray): batch of label waypoints [batch_size, T, 4] or [batch_size, horizon, 2]
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        normalized (bool): whether the waypoints are normalized
        save_folder (str): folder to save the images. If None, will not save the images
        epoch (int): current epoch number
        num_images_preds (int): number of images to visualize
        use_mlflow (bool): whether to use mlflow to log the images
        display (bool): whether to display the images
    """
    visualize_path = None
    if save_folder is not None:
        visualize_path = os.path.join(
            save_folder, "visualize", eval_type, f"epoch{epoch}", "action_prediction"
        )

    if not os.path.exists(visualize_path):
        os.makedirs(visualize_path)

    # Check if batch_pred_waypoints has multiple predictions per sample
    multi_pred = False
    if len(batch_pred_waypoints) > len(batch_obs_images):
        multi_pred = True

    # Only check that the other dimensions match
    assert (
        len(batch_obs_images)
        == len(batch_goal_images)
        == len(batch_goals)
        == len(batch_label_waypoints)
    )

    # Only enforce matching batch size for pred_waypoints if not multi_pred
    if not multi_pred:
        assert len(batch_pred_waypoints) == len(batch_obs_images)

    dataset_names = list(data_config.keys())
    dataset_names.sort()

    batch_size = batch_obs_images.shape[0]
    for i in range(min(batch_size, num_images_preds)):
        obs_img = numpy_to_img(batch_obs_images[i])
        goal_img = numpy_to_img(batch_goal_images[i])
        dataset_name = dataset_names[int(dataset_indices[i])]
        goal_pos = batch_goals[i]
        label_waypoints = batch_label_waypoints[i]

        # Handle multiple predictions per sample
        if multi_pred:
            # If we have multiple predictions, select 10 random ones to visualize
            import random
            num_preds_to_show = min(2, len(batch_pred_waypoints))
            pred_indices = random.sample(range(len(batch_pred_waypoints)), num_preds_to_show)
            pred_waypoints = [batch_pred_waypoints[idx] for idx in pred_indices]
        else:
            # Single prediction per sample
            pred_waypoints = [batch_pred_waypoints[i]]

        if normalized:
            # Apply normalization to all predictions
            if multi_pred:
                for j in range(len(pred_waypoints)):
                    pred_waypoints[j] = pred_waypoints[j] * data_config[dataset_name]["metric_waypoint_spacing"]
            else:
                pred_waypoints[0] = pred_waypoints[0] * data_config[dataset_name]["metric_waypoint_spacing"]

            label_waypoints *= data_config[dataset_name]["metric_waypoint_spacing"]
            goal_pos *= data_config[dataset_name]["metric_waypoint_spacing"]

        save_path = None
        if visualize_path is not None:
            # Include batch index and epoch in filename to avoid overwriting
            save_path = os.path.join(visualize_path, f"epoch{epoch}_batch{str(i).zfill(4)}_sample{str(i).zfill(4)}.png")

        # Modified to handle multiple predictions
        compare_waypoints_pred_to_label(
            obs_img,
            goal_img,
            dataset_name,
            goal_pos,
            pred_waypoints,  # Now a list of predictions
            label_waypoints,
            save_path,
            display,
            multi_pred=multi_pred,  # Pass flag to indicate multiple predictions
        )
        if use_mlflow:
            mlflow.log_artifact(save_path, artifact_path=f"{eval_type}/action_prediction")


def compare_waypoints_pred_to_label(
    obs_img,
    goal_img,
    dataset_name: str,
    goal_pos: np.ndarray,
    pred_waypoints: np.ndarray,
    label_waypoints: np.ndarray,
    save_path: Optional[str] = None,
    display: Optional[bool] = False,
    multi_pred: bool = False,
):
    """
    Compare predicted path with the gt path of waypoints using egocentric visualization.

    Args:
        obs_img: image of the observation
        goal_img: image of the goal
        dataset_name: name of the dataset found in data_config.yaml (e.g. "recon")
        goal_pos: goal position in the image
        pred_waypoints: predicted waypoints in the image
        label_waypoints: label waypoints in the image
        save_path: path to save the figure
        display: whether to display the figure
    """

    fig, ax = plt.subplots(1, 3)
    start_pos = np.array([0, 0])
    # Handle multiple predictions
    if multi_pred:
        # pred_waypoints is a list of predictions
        trajs = pred_waypoints + [label_waypoints]
    else:
        # Single prediction
        if len(pred_waypoints.shape) > 2:
            trajs = [*pred_waypoints, label_waypoints]
        else:
            trajs = [pred_waypoints, label_waypoints]
    # Create colors for multiple trajectories if needed
    if multi_pred:
        # Use different shades of cyan for predictions and yellow for ground truth
        # CYAN and YELLOW are numpy arrays, so we need to create a list of arrays
        YELLOW = np.array([1, 1, 0])  # RGB for yellow
        traj_colors = [np.array(CYAN) for _ in range(len(pred_waypoints))] + [np.array(YELLOW)]

        # Use different alphas for predictions
        # Make sure we have one alpha for each trajectory (predictions + ground truth)
        num_trajs = len(pred_waypoints) + 1  # +1 for ground truth

        # Create alphas: lower values for most predictions, higher for last prediction and ground truth
        if num_trajs == 2:  # One prediction, one ground truth
            traj_alphas = [0.7, 1.0]
        else:  # Multiple predictions plus ground truth
            traj_alphas = [0.3] * (num_trajs - 2) + [0.7, 1.0]
    else:
        YELLOW = np.array([1, 1, 0])  # RGB for yellow
        traj_colors = [np.array(CYAN), np.array(YELLOW)]
        traj_alphas = None

    plot_trajs_and_points(
        ax[0],
        trajs,
        [start_pos, goal_pos],
        traj_colors=traj_colors,
        point_colors=[GREEN, RED],
        traj_alphas=traj_alphas,
    )

    plot_trajs_and_points_on_image(
        ax[1],
        obs_img,
        dataset_name,
        trajs,
        [start_pos, goal_pos],
        traj_colors=traj_colors,
        point_colors=[GREEN, RED],
        traj_alphas=traj_alphas,
    )
    ax[2].imshow(goal_img)

    fig.set_size_inches(18.5, 10.5)
    ax[0].set_title(f"Action Prediction")
    ax[1].set_title(f"Observation")
    ax[2].set_title(f"Goal")

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
        )

    if not display:
        plt.close(fig)


def plot_trajs_and_points_on_image(
    ax: plt.Axes,
    img: np.ndarray,
    dataset_name: str,
    list_trajs: list,
    list_points: list,
    traj_colors: list = [CYAN, MAGENTA],
    point_colors: list = [RED, GREEN],
    traj_alphas: Optional[list] = None,
):
    """
    Plot trajectories and points on an image. If there is no configuration for the camera interinstics of the dataset, the image will be plotted as is.
    Args:
        ax: matplotlib axis
        img: image to plot
        dataset_name: name of the dataset found in data_config.yaml (e.g. "recon")
        list_trajs: list of trajectories, each trajectory is a numpy array of shape (horizon, 2) (if there is no yaw) or (horizon, 4) (if there is yaw)
        list_points: list of points, each point is a numpy array of shape (2,)
        traj_colors: list of colors for trajectories
        point_colors: list of colors for points
    """
    assert len(list_trajs) <= len(traj_colors), "Not enough colors for trajectories"
    assert len(list_points) <= len(point_colors), "Not enough colors for points"
    assert (
        dataset_name in data_config
    ), f"Dataset {dataset_name} not found in data/data_config.yaml"

    ax.imshow(img)
    if (
        "camera_metrics" in data_config[dataset_name]
        and "camera_height" in data_config[dataset_name]["camera_metrics"]
        and "camera_matrix" in data_config[dataset_name]["camera_metrics"]
        and "dist_coeffs" in data_config[dataset_name]["camera_metrics"]
    ):
        camera_height = data_config[dataset_name]["camera_metrics"]["camera_height"]
        camera_x_offset = data_config[dataset_name]["camera_metrics"]["camera_x_offset"]

        fx = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["fx"]
        fy = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["fy"]
        cx = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["cx"]
        cy = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["cy"]
        camera_matrix = gen_camera_matrix(fx, fy, cx, cy)

        k1 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k1"]
        k2 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k2"]
        p1 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["p1"]
        p2 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["p2"]
        k3 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k3"]
        dist_coeffs = np.array([k1, k2, p1, p2, k3, 0.0, 0.0, 0.0])

        for i, traj in enumerate(list_trajs):
            xy_coords = traj[:, :2]  # (horizon, 2)
            traj_pixels = get_pos_pixels(
                xy_coords,
                camera_height,
                camera_x_offset,
                camera_matrix,
                dist_coeffs,
                clip=False,
            )
            if len(traj_pixels.shape) == 2:
                # Make sure we don't try to access an index that's out of range
                if i < len(traj_colors):
                    color = traj_colors[i]
                else:
                    color = traj_colors[0] if len(traj_colors) > 0 else CYAN

                alpha = traj_alphas[i] if traj_alphas is not None and i < len(traj_alphas) else 1.0

                ax.plot(
                    traj_pixels[:250, 0],
                    traj_pixels[:250, 1],
                    color=color,
                    alpha=alpha,
                    lw=2.5,
                )

        for i, point in enumerate(list_points):
            if len(point.shape) == 1:
                # add a dimension to the front of point
                point = point[None, :2]
            else:
                point = point[:, :2]
            pt_pixels = get_pos_pixels(
                point,
                camera_height,
                camera_x_offset,
                camera_matrix,
                dist_coeffs,
                clip=True,
            )
            ax.plot(
                pt_pixels[:250, 0],
                pt_pixels[:250, 1],
                color=point_colors[i],
                marker="o",
                markersize=10.0,
            )
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlim((0.5, VIZ_IMAGE_SIZE[0] - 0.5))
        ax.set_ylim((VIZ_IMAGE_SIZE[1] - 0.5, 0.5))


def plot_trajs_and_points(
    ax: plt.Axes,
    list_trajs: list,
    list_points: list,
    traj_colors: list = [CYAN, MAGENTA],
    point_colors: list = [RED, GREEN],
    traj_labels: Optional[list] = ["prediction", "ground truth"],
    point_labels: Optional[list] = ["robot", "goal"],
    traj_alphas: Optional[list] = None,
    point_alphas: Optional[list] = None,
    quiver_freq: int = 1,
    default_coloring: bool = True,
):
    """
    Plot trajectories and points that could potentially have a yaw.

    Args:
        ax: matplotlib axis
        list_trajs: list of trajectories, each trajectory is a numpy array of shape (horizon, 2) (if there is no yaw) or (horizon, 4) (if there is yaw)
        list_points: list of points, each point is a numpy array of shape (2,)
        traj_colors: list of colors for trajectories
        point_colors: list of colors for points
        traj_labels: list of labels for trajectories
        point_labels: list of labels for points
        traj_alphas: list of alphas for trajectories
        point_alphas: list of alphas for points
        quiver_freq: frequency of quiver plot (if the trajectory data includes the yaw of the robot)
    """
    assert (
        len(list_trajs) <= len(traj_colors) or default_coloring
    ), "Not enough colors for trajectories"
    assert len(list_points) <= len(point_colors), "Not enough colors for points"

    assert (
        traj_labels is None or len(list_trajs) == len(traj_labels) or default_coloring
    ), "Not enough labels for trajectories"

    assert point_labels is None or len(list_points) == len(
        point_labels
    ), "Not enough labels for points"

    for i, traj in enumerate(list_trajs):
        if traj_labels is None:
            # Make sure we don't try to access an index that's out of range
            if i < len(traj_colors):
                color = traj_colors[i]
            else:
                color = traj_colors[0] if len(traj_colors) > 0 else CYAN

            alpha = traj_alphas[i] if traj_alphas is not None and i < len(traj_alphas) else 1.0

            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=color,
                alpha=alpha,
                marker="o",
            )
        else:
            # Make sure we don't try to access an index that's out of range
            if i < len(traj_colors):
                color = traj_colors[i]
            else:
                color = traj_colors[0] if len(traj_colors) > 0 else CYAN

            label = traj_labels[i] if i < len(traj_labels) else "trajectory"
            alpha = traj_alphas[i] if traj_alphas is not None and i < len(traj_alphas) else 1.0

            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=color,
                label=label,
                alpha=alpha,
                marker="o",
            )

        if (
            traj.shape[1] > 2 and quiver_freq > 0
        ):  # traj data also includes yaw of the robot
            bearings = gen_bearings_from_waypoints(traj)
            # Make sure we don't try to access an index that's out of range
            if i < len(traj_colors):
                color = traj_colors[i]
            else:
                color = traj_colors[0] if len(traj_colors) > 0 else CYAN
            ax.quiver(
                traj[::quiver_freq, 0],
                traj[::quiver_freq, 1],
                bearings[::quiver_freq, 0],
                bearings[::quiver_freq, 1],
                color=color * 0.5,
                scale=1.0,
            )
    for i, pt in enumerate(list_points):
        if point_labels is None:
            ax.plot(
                pt[0],
                pt[1],
                color=point_colors[i],
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0,
            )
        else:
            ax.plot(
                pt[0],
                pt[1],
                color=point_colors[i],
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0,
                label=point_labels[i],
            )

    # put the legend below the plot
    if traj_labels is not None or point_labels is not None:
        ax.legend()
        ax.legend(bbox_to_anchor=(0.0, -0.5), loc="upper left", ncol=2)
    ax.set_aspect("equal", "box")


def angle_to_unit_vector(theta):
    """Converts an angle to a unit vector."""
    return np.array([np.cos(theta), np.sin(theta)])


def gen_bearings_from_waypoints(
    waypoints: np.ndarray,
    mag=0.2,
) -> np.ndarray:
    """Generate bearings from waypoints, (x, y, sin(theta), cos(theta))."""
    bearing = []
    for i in range(0, len(waypoints)):
        if waypoints.shape[1] > 3:  # label is sin/cos repr
            v = waypoints[i, 2:]
            # normalize v
            v = v / np.linalg.norm(v)
            v = v * mag
        else:  # label is radians repr
            v = mag * angle_to_unit_vector(waypoints[i, 2])
        bearing.append(v)
    bearing = np.array(bearing)
    return bearing


def project_points(
    xy: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.

    Args:
        xy: array of shape (batch_size, horizon, 2) representing (x, y) coordinates
        camera_height: height of the camera above the ground (in meters)
        camera_x_offset: offset of the camera from the center of the car (in meters)
        camera_matrix: 3x3 matrix representing the camera's intrinsic parameters
        dist_coeffs: vector of distortion coefficients


    Returns:
        uv: array of shape (batch_size, horizon, 2) representing (u, v) coordinates on the 2D image plane
    """
    batch_size, horizon, _ = xy.shape

    # create 3D coordinates with the camera positioned at the given height
    xyz = np.concatenate(
        [xy, -camera_height * np.ones(list(xy.shape[:-1]) + [1])], axis=-1
    )

    # create dummy rotation and translation vectors
    rvec = tvec = (0, 0, 0)

    xyz[..., 0] += camera_x_offset
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    uv, _ = cv2.projectPoints(
        xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist_coeffs
    )
    uv = uv.reshape(batch_size, horizon, 2)

    return uv


def get_pos_pixels(
    points: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    clip: Optional[bool] = False,
):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.
    Args:
        points: array of shape (batch_size, horizon, 2) representing (x, y) coordinates
        camera_height: height of the camera above the ground (in meters)
        camera_x_offset: offset of the camera from the center of the car (in meters)
        camera_matrix: 3x3 matrix representing the camera's intrinsic parameters
        dist_coeffs: vector of distortion coefficients

    Returns:
        pixels: array of shape (batch_size, horizon, 2) representing (u, v) coordinates on the 2D image plane
    """
    pixels = project_points(
        points[np.newaxis], camera_height, camera_x_offset, camera_matrix, dist_coeffs
    )[0]
    pixels[:, 0] = VIZ_IMAGE_SIZE[0] - pixels[:, 0]
    if clip:
        pixels = np.array(
            [
                [
                    np.clip(p[0], 0, VIZ_IMAGE_SIZE[0]),
                    np.clip(p[1], 0, VIZ_IMAGE_SIZE[1]),
                ]
                for p in pixels
            ]
        )
    else:
        pixels = np.array(
            [
                p
                for p in pixels
                if np.all(p > 0) and np.all(p < [VIZ_IMAGE_SIZE[0], VIZ_IMAGE_SIZE[1]])
            ]
        )
    return pixels


def gen_camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Args:
        fx: focal length in x direction
        fy: focal length in y direction
        cx: principal point x coordinate
        cy: principal point y coordinate
    Returns:
        camera matrix
    """
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
