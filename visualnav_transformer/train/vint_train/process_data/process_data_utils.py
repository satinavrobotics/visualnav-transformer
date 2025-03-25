import io
from typing import Any, Dict, List, Tuple
import os
import cv2
import numpy as np
import rosbag
import torchvision.transforms.functional as TF
from PIL import Image
from sensor_msgs.msg import PointCloud2, Imu, NavSatFix, LaserScan
import sensor_msgs.point_cloud2 as pc2
import json
from datetime import datetime

IMAGE_SIZE = (160, 120)
IMAGE_ASPECT_RATIO = 4 / 3

def timestamp_to_datestr(timestamp):
    if hasattr(timestamp, "to_sec"):
        timestamp = timestamp.to_sec()
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H-%M-%S")

def process_images(im_list: List, img_process_func) -> List:
    """
    Process image data from a topic that publishes ros images into a list of PIL images
    """
    images = []
    for img_msg in im_list:
        img = img_process_func(img_msg)
        images.append(img)
    return images


def process_tartan_img(msg) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/Image to a PIL image for the tartan_drive dataset
    """
    img = ros_to_numpy(msg, output_resolution=IMAGE_SIZE) * 255
    img = img.astype(np.uint8)
    # reverse the axis order to get the image in the right orientation
    img = np.moveaxis(img, 0, -1)
    # convert rgb to bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)
    return img


def process_locobot_img(msg) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/Image to a PIL image for the locobot dataset
    """
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    pil_image = Image.fromarray(img)
    return pil_image


def process_scand_img(msg) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/CompressedImage to a PIL image for the scand dataset
    """
    # convert sensor_msgs/CompressedImage to PIL image
    img = Image.open(io.BytesIO(msg.data))
        
    # center crop image to 4:3 aspect ratio
    w, h = img.size
    img = TF.center_crop(
        img, (h, int(h * IMAGE_ASPECT_RATIO))
    )  # crop to the right ratio
    # resize image to IMAGE_SIZE
    img = img.resize(IMAGE_SIZE)
    return img


############## Add custom image processing functions here #############


def process_sacson_img(msg) -> Image:
    np_arr = np.fromstring(msg.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_np)
    return pil_image


#######################################################################


def process_odom(
    odom_list: List,
    odom_process_func: Any,
    ang_offset: float = 0.0,
) -> Dict[np.ndarray, np.ndarray]:
    """
    Process odom data from a topic that publishes nav_msgs/Odometry into position and yaw
    """
    xys = []
    yaws = []
    for odom_msg in odom_list:
        xy, yaw = odom_process_func(odom_msg, ang_offset)
        xys.append(xy)
        yaws.append(yaw)
    return {"position": np.array(xys), "yaw": np.array(yaws)}


def nav_to_xy_yaw(odom_msg, ang_offset: float) -> Tuple[List[float], float]:
    """
    Process odom data from a topic that publishes nav_msgs/Odometry into position
    """

    position = odom_msg.pose.pose.position
    orientation = odom_msg.pose.pose.orientation
    yaw = (
        quat_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w)
        + ang_offset
    )
    return [position.x, position.y], yaw


############ Add custom odometry processing functions here ############
def save_pointcloud_pcd(point_list, output_dir, index: int, timestamp):
    """
    Saves a single frame of PointCloud2 data as a PCD file into a directory.

    Args:
        point_list: List of (x, y, z) points.
        output_dir: Directory to save the file.
        index: Frame index for naming.
        timestamp: ROS timestamp (rospy.Time or float).
    """
    
    if point_list is None or len(point_list) == 0:
        # print("⚠️ Skipping empty PointCloud2 frame.")
        return
    
    os.makedirs(output_dir, exist_ok=True)

    dt_str = timestamp_to_datestr(timestamp)
    filename = f"pc_{index:06d}_{dt_str}.pcd"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {len(point_list)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(point_list)}\n")
        f.write("DATA ascii\n")

        for p in point_list:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

    # print(f"✅ Saved PointCloud2 frame: {filepath}")


def save_json(data, filepath):
    """
    Saves data to a JSON file if data exists.
    """
    if data:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved {filepath}")


def process_laserscan(msg: LaserScan, output_dir: str, index: int, timestamp):
    """
    Convert a LaserScan message to a 3D PCD file and save it with a timestamp-based name.

    Args:
        msg (LaserScan): The ROS LaserScan message.
        output_dir (str): Directory where the file should be saved.
        index (int): Frame index.
        timestamp: ROS bag timestamp (float or rospy.Time).
    """
    if not msg.ranges:
        print("⚠️ Skipping empty LaserScan frame.")
        return
    
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{index:06d}_{timestamp.to_sec()}.pcd"
    pcd_filepath = os.path.join(output_dir, filename)

    angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
    points = [
        (
            r * np.cos(angle), 
            r * np.sin(angle),
            0.0  # LIDAR is 2D, so Z = 0
        )
        for r, angle in zip(msg.ranges, angles)
        if r > msg.range_min and r < msg.range_max
    ]

    with open(pcd_filepath, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")

        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

    # print(f"✅ Saved LaserScan frame: {pcd_filepath}")


def process_imu(msg) -> Dict[str, Any] | None:
    """Process IMU data and return as a dictionary, or None if empty/invalid."""
    try:
        vals = [
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w,
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        ]
        is_valid = any(not np.isnan(v) and abs(v) > 1e-6 for v in vals)
    except AttributeError:
        return None  # Missing fields

    if not is_valid:
        return None  # All values zero or NaN

    return {
        "timestamp": msg.header.stamp.to_sec(),
        "orientation": {
            "x": msg.orientation.x,
            "y": msg.orientation.y,
            "z": msg.orientation.z,
            "w": msg.orientation.w,
        },
        "angular_velocity": {
            "x": msg.angular_velocity.x,
            "y": msg.angular_velocity.y,
            "z": msg.angular_velocity.z,
        },
        "linear_acceleration": {
            "x": msg.linear_acceleration.x,
            "y": msg.linear_acceleration.y,
            "z": msg.linear_acceleration.z,
        },
    }


def process_gps(msg: NavSatFix) -> Dict[str, Any]:
    """Process GPS data and return as a dictionary."""
    return {
        "timestamp": msg.header.stamp.to_sec() if hasattr(msg, "header") else None,
        "latitude": msg.latitude,
        "longitude": msg.longitude,
        "altitude": msg.altitude,
    }
    
def process_joint_states(msg):
    return {
        "name": list(msg.name),
        "position": list(msg.position),
        "velocity": list(msg.velocity),
        "effort": list(msg.effort),
        "timestamp": msg.header.stamp.to_sec()
    }

def process_joystick(msg):
    axes = list(msg.axes)
    buttons = list(msg.buttons)
    is_active = any(abs(a) > 1e-3 for a in axes) or any(b != 0 for b in buttons)
    if not is_active:
        return None

    return {
        "axes": axes,
        "buttons": buttons,
        "timestamp": msg.header.stamp.to_sec()
    }

def process_cmd_vel(msg, timestamp):
    return {
        "linear": {
            "x": msg.linear.x,
            "y": msg.linear.y,
            "z": msg.linear.z,
        },
        "angular": {
            "x": msg.angular.x,
            "y": msg.angular.y,
            "z": msg.angular.z,
        },
        "timestamp": timestamp  # Use timestamp from rosbag.read_messages()
    }

def process_camera_info(msg):
    """
    Converts a ROS sensor_msgs/CameraInfo message into a JSON-serializable dictionary.
    """
    return {
        "height": msg.height,
        "width": msg.width,
        "distortion_model": msg.distortion_model,
        "D": list(msg.D),  # Distortion coefficients
        "K": list(msg.K),  # Camera intrinsic matrix
        "R": list(msg.R),  # Rectification matrix
        "P": list(msg.P),  # Projection matrix
        "binning_x": msg.binning_x,
        "binning_y": msg.binning_y,
        "roi": {
            "x_offset": msg.roi.x_offset,
            "y_offset": msg.roi.y_offset,
            "height": msg.roi.height,
            "width": msg.roi.width,
            "do_rectify": msg.roi.do_rectify,
        }
    }
    
def process_tf_message(msg):
    return [
        {
            "timestamp": tf.header.stamp.to_sec(),
            "frame_id": tf.header.frame_id,
            "child_frame_id": tf.child_frame_id,
            "transform": {
                "translation": {
                    "x": tf.transform.translation.x,
                    "y": tf.transform.translation.y,
                    "z": tf.transform.translation.z,
                },
                "rotation": {
                    "x": tf.transform.rotation.x,
                    "y": tf.transform.rotation.y,
                    "z": tf.transform.rotation.z,
                    "w": tf.transform.rotation.w,
                },
            },
        }
        for tf in msg.transforms
    ]


def process_gnss_extra(msg, topic, timestamp=None):
    result = {"timestamp": timestamp.to_sec() if timestamp else None}

    if topic.endswith("time_reference"):
        if hasattr(msg, "header") and hasattr(msg, "time_ref") and hasattr(msg, "source"):
            result.update({
                "time_ref": msg.time_ref.to_sec() if hasattr(msg.time_ref, "to_sec") else None
            })
            return result

    elif topic.endswith("vel"):
        if hasattr(msg, "header") and hasattr(msg, "twist"):
            result["linear"] = {
                "x": msg.twist.linear.x,
                "y": msg.twist.linear.y,
                "z": msg.twist.linear.z,
            }
            result["angular"] = {
                "x": msg.twist.angular.x,
                "y": msg.twist.angular.y,
                "z": msg.twist.angular.z,
            }
            return result

    # If we can't parse the msg, return None
    # print(f"⚠️ Unknown or malformed GNSS extra message on topic: {topic}")
    return None
    
#######################################################################


def get_images_and_odom(
    bag: rosbag.Bag,
    imtopics: List[str],
    odomtopics: List[str],
    img_process_func: Any,
    odom_process_func: Any,
    rate: float = 4.0,
    ang_offset: float = 0.0,
):
    """
    Get image and odom data from a bag file, processing only available image topics.

    Args:
        bag (rosbag.Bag): bag file
        imtopics (list[str]): list of topic names for image data
        odomtopics (list[str]): list of topic names for odom data
        img_process_func (Any): function to process image data
        odom_process_func (Any): function to process odom data
        rate (float, optional): rate to sample data. Defaults to 4.0.
        ang_offset (float, optional): angle offset to add to odom data. Defaults to 0.0.

    Returns:
        img_data_dict (dict): Dictionary of {topic_name: list of PIL images}
        traj_data (dict): Dictionary with position and yaw data
    """

    # Print requested topics from config
    print(f"Config-specified image topics: {imtopics}")

    # Check available topics in the ROS bag
    available_imtopics = [t for t in imtopics if bag.get_message_count(t) > 0]
    available_odomtopics = [t for t in odomtopics if bag.get_message_count(t) > 0]

    print(f"Available image topics in the bag: {available_imtopics}")

    # Print message counts for all topics
    for topic in imtopics:
        print(f"Topic {topic} has {bag.get_message_count(topic)} messages")

    if not available_imtopics or not available_odomtopics:
        print(f"Skipping bag {bag.filename} - Missing required topics.")
        return None, None

    print(f"Processing {bag.filename} with topics: {available_imtopics}")

    # **Initialize storage for synced data**
    synced_imdata = {topic: [] for topic in available_imtopics}
    synced_odomdata = []

    currtime = bag.get_start_time()
    curr_imdata = {topic: None for topic in available_imtopics}
    curr_odomdata = None

    # **Read messages from available topics**
    for topic, msg, t in bag.read_messages(topics=available_imtopics + available_odomtopics):
        if topic in available_imtopics:
            curr_imdata[topic] = msg
        elif topic in available_odomtopics:
            curr_odomdata = msg

        if (t.to_sec() - currtime) >= 1.0 / rate:
            if curr_odomdata is not None and all(curr_imdata.values()):
                for topic in available_imtopics:
                    synced_imdata[topic].append(curr_imdata[topic])
                synced_odomdata.append(curr_odomdata)
                currtime = t.to_sec()

    # **Process image and odometry data**
    img_data_dict = {topic: process_images(synced_imdata[topic], img_process_func) for topic in available_imtopics}
    traj_data = process_odom(synced_odomdata, odom_process_func, ang_offset=ang_offset)

    return img_data_dict, traj_data


def is_backwards(
    pos1: np.ndarray, yaw1: float, pos2: np.ndarray, eps: float = 1e-5
) -> bool:
    """
    Check if the trajectory is going backwards given the position and yaw of two points
    Args:
        pos1: position of the first point

    """
    dx, dy = pos2 - pos1
    return dx * np.cos(yaw1) + dy * np.sin(yaw1) < eps


# cut out non-positive velocity segments of the trajectory
def filter_backwards(
    img_dict: Dict[str, List[Image.Image]],
    traj_data: Dict[str, np.ndarray],
    start_slack: int = 0,
    end_slack: int = 0,
):
    """
    Filter out backward-moving segments while handling missing topics gracefully.

    Args:
        img_dict (dict): Dictionary of {topic_name: list of images}
        traj_data (dict): Dictionary with "position" and "yaw"
        start_slack (int, optional): Points to ignore at the start. Defaults to 0.
        end_slack (int, optional): Points to ignore at the end. Defaults to 0.

    Returns:
        cut_trajs (dict): Dictionary {topic_name: (list of images, trajectory data)}
    """

    traj_pos = traj_data.get("position", [])
    traj_yaws = traj_data.get("yaw", [])

    if len(traj_pos) == 0 or len(traj_yaws) == 0:
        print("Skipping filtering - No trajectory data available.")
        return {}

    cut_trajs = {}

    for topic, img_list in img_dict.items():
        if len(img_list) == 0:
            print(f"Skipping topic {topic} - No images found.")
            continue

        cut_trajs[topic] = []
        new_traj_pairs = []

        for i in range(1, len(traj_pos) - end_slack):
            pos1, pos2 = traj_pos[i - 1], traj_pos[i]
            yaw1 = traj_yaws[i - 1]

            if not is_backwards(pos1, yaw1, pos2):
                if not new_traj_pairs:
                    new_traj_pairs.append((img_list[i - 1], [*traj_pos[i - 1], traj_yaws[i - 1]]))
                elif i == len(traj_pos) - end_slack - 1:
                    cut_trajs[topic].append(process_pair(new_traj_pairs))
                else:
                    new_traj_pairs.append((img_list[i - 1], [*traj_pos[i - 1], traj_yaws[i - 1]]))
            elif new_traj_pairs:
                cut_trajs[topic].append(process_pair(new_traj_pairs))
                new_traj_pairs = []

    return cut_trajs

def process_pair(traj_pair: list) -> Tuple[List[Image.Image], Dict[str, np.ndarray]]:
    """
    Process a list of (image, trajectory) pairs into separate lists.
    
    Args:
        traj_pair: List of tuples [(image, trajectory_data)]
    
    Returns:
        (image_list, trajectory_dict)
    """
    new_img_list, new_traj_data = zip(*traj_pair)
    new_traj_data = np.array(new_traj_data)
    
    return list(new_img_list), {"position": new_traj_data[:, :2], "yaw": new_traj_data[:, 2]}


def quat_to_yaw(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Convert a batch quaternion into a yaw angle
    yaw is rotation around z in radians (counterclockwise)
    """
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return yaw


def ros_to_numpy(
    msg, nchannels=3, empty_value=None, output_resolution=None, aggregate="none"
):
    """
    Convert a ROS image message to a numpy array
    """
    if output_resolution is None:
        output_resolution = (msg.width, msg.height)

    is_rgb = "8" in msg.encoding
    if is_rgb:
        data = np.frombuffer(msg.data, dtype=np.uint8).copy()
    else:
        data = np.frombuffer(msg.data, dtype=np.float32).copy()

    data = data.reshape(msg.height, msg.width, nchannels)

    if empty_value:
        mask = np.isclose(abs(data), empty_value)
        fill_value = np.percentile(data[~mask], 99)
        data[mask] = fill_value

    data = cv2.resize(
        data,
        dsize=(output_resolution[0], output_resolution[1]),
        interpolation=cv2.INTER_AREA,
    )

    if aggregate == "littleendian":
        data = sum([data[:, :, i] * (256**i) for i in range(nchannels)])
    elif aggregate == "bigendian":
        data = sum([data[:, :, -(i + 1)] * (256**i) for i in range(nchannels)])

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    else:
        data = np.moveaxis(data, 2, 0)  # Switch to channels-first

    if is_rgb:
        data = data.astype(np.float32) / (
            255.0 if aggregate == "none" else 255.0**nchannels
        )

    return data

# Convert NumPy arrays to lists before saving
def numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy array to list
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}  # Convert recursively
    elif isinstance(obj, list):
        return [numpy_to_list(v) for v in obj]  # Convert lists of NumPy arrays
    return obj  # Return the object if it's neither dict nor ndarray
