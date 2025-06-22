import argparse
import os
import rosbag
import tqdm
import yaml
import json
from collections import defaultdict

# utils
from visualnav_transformer.train.vint_train.process_data.process_data_utils import *

def process_and_save_additional_data(bag, config, bag_folder):
    """
    Processes and saves additional data types such as PointCloud2, IMU, GPS, LaserScan, TF, and GNSS extras.
    """
    # Data storage
    imu_data, gps_data = [], []
    joint_states_data, joystick_data, cmd_vel_data = [], [], []
    tf_data = []
    gnss_extra_data = defaultdict(list)
    camera_info_data = defaultdict(dict)

    # Folders for pointcloud and laserscan
    laserscan_folder = os.path.join(bag_folder, "laserscan")
    pointcloud_folder = os.path.join(bag_folder, "pointcloud")

    pointcloud_idx = 0
    laserscan_idx = 0

    for topic, msg, timestamp in bag.read_messages():
        if topic in config.get("pointcloud_topics", []):
            point_list = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            save_pointcloud_pcd(point_list, pointcloud_folder, pointcloud_idx, timestamp)
            pointcloud_idx += 1

        elif topic in config.get("lidar_topics", []):
            process_laserscan(msg, laserscan_folder, laserscan_idx, timestamp)
            laserscan_idx += 1

        elif topic in config.get("imu_topics", []):
            imu_entry = process_imu(msg)
            if imu_entry:
                imu_data.append(imu_entry)

        elif topic in config.get("gps_topics", []):
            gps_data.append(process_gps(msg))

        elif topic in config.get("joint_states_topics", []):
            joint_states_data.append(process_joint_states(msg))

        elif topic in config.get("joystick_topics", []):
            joystick_data.append(process_joystick(msg))

        elif topic in config.get("cmd_vel_topics", []):
            cmd_vel_data.append(process_cmd_vel(msg, timestamp.to_sec()))

        elif topic in config.get("camera_info_topics", []):
            image_topic_guess = topic.replace("/camera_info", "/image/compressed")
            camera_info_data[image_topic_guess] = process_camera_info(msg)

        elif topic in config.get("tf_topics", []):
            tf_data.extend(process_tf_message(msg))

        elif topic in config.get("gnss_extra_topics", []):
            parsed = process_gnss_extra(msg, topic, timestamp)
            if parsed:
                gnss_extra_data[topic].append(parsed)

    # Save all sensor data
    save_json(imu_data, os.path.join(bag_folder, "imu.json"))
    save_json(gps_data, os.path.join(bag_folder, "gps.json"))
    save_json(joint_states_data, os.path.join(bag_folder, "joint_states.json"))
    save_json(joystick_data, os.path.join(bag_folder, "joystick.json"))
    save_json(cmd_vel_data, os.path.join(bag_folder, "cmd_vel.json"))
    save_json(tf_data, os.path.join(bag_folder, "tf.json"))

    # Save each GNSS extra topic separately
    for gnss_topic, entries in gnss_extra_data.items():
        topic_name = gnss_topic.replace("/", "_")
        save_json(entries, os.path.join(bag_folder, f"{topic_name}.json"))

    return camera_info_data


def main(args: argparse.Namespace):

    # load the config file
    with open("/app/visualnav-transformer/visualnav_transformer/train/vint_train/process_data/process_bags_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # create output dir if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # iterate recurisively through all the folders and get the path of files with .bag extension in the args.input_dir
    if os.path.isfile(args.input_dir) and args.input_dir.endswith(".bag"):
        bag_files = [args.input_dir]
    else:
        bag_files = []
        for root, dirs, files in os.walk(args.input_dir):
            for file in files:
                if file.endswith(".bag"):
                    bag_files.append(os.path.join(root, file))
        if args.num_trajs >= 0:
            bag_files = bag_files[: args.num_trajs]

    # processing loop
    for bag_path in tqdm.tqdm(bag_files, desc="Bags processed"):
        try:
            b = rosbag.Bag(bag_path)
        except rosbag.ROSBagException as e:
            print(e)
            print(f"Error loading {bag_path}. Skipping...")
            continue

        # **Create a folder for each bag inside the output directory**
        bag_name = os.path.splitext(os.path.basename(bag_path))[0]  # Extract bag name
        bag_folder = os.path.join(args.output_dir, bag_name)
        os.makedirs(bag_folder, exist_ok=True)

        # load the hdf5 file
        bag_img_data, bag_traj_data = get_images_and_odom(
            b,
            config[args.dataset_name]["imtopics"],
            config[args.dataset_name]["odomtopics"],
            eval(config[args.dataset_name]["img_process_func"]),
            eval(config[args.dataset_name]["odom_process_func"]),
            rate=args.sample_rate,
            ang_offset=config[args.dataset_name]["ang_offset"],
        )

        if bag_img_data is None or bag_traj_data is None:
            print(
                f"{bag_path} did not have the topics we were looking for. Skipping..."
            )
            continue
        
        # Remove backwards movement
        cut_trajs = filter_backwards(bag_img_data, bag_traj_data)
        # **Process and save additional sensor data**
        camera_info_data = process_and_save_additional_data(b, config[args.dataset_name], bag_folder)

        # Process each topic separately
        for topic, traj_segments in cut_trajs.items():
            topic_clean = topic.replace("/", "_")  # Ensure valid folder names
            topic_folder = os.path.join(bag_folder, topic_clean)
            os.makedirs(topic_folder, exist_ok=True)  # Create separate folder for each topic

            for seg_idx, (img_data_i, traj_data_i) in enumerate(traj_segments):
                if not img_data_i:
                    print(f"Skipping {topic} for segment {seg_idx} (No images found)")
                    continue

                # Save images
                for idx, img in enumerate(img_data_i):
                    if "right" in topic:
                        img = img.rotate(180, expand=True)
                    img.save(os.path.join(topic_folder, f"{idx}.jpg"))
                    
            # topic is the actual image topic here
            if topic in camera_info_data:
                cam_info_path = os.path.join(topic_folder, "camera_info.json")
                with open(cam_info_path, "w") as f:
                    json.dump(camera_info_data[topic], f, indent=4)
            else:
                print(f"⚠️ No camera_info found for {topic}")

        # **Save trajectory data in the root of bag folder**
        traj_data_path = os.path.join(bag_folder, "traj_data.json")
        with open(traj_data_path, "w") as f:
            json.dump(numpy_to_list(bag_traj_data), f, indent=4)

        b.close()
            
        #'''
        # After processing the bag, delete it to free up space.
        try:
            os.remove(bag_path)
            print(f"Deleted processed bag: {bag_path}")
        except Exception as e:
            print(f"Error deleting {bag_path}: {e}")
        #'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # get arguments for the recon input dir and the output dir
    # add dataset name
    parser.add_argument(
        "--dataset-name",
        "-d",
        type=str,
        help="name of the dataset (must be in process_config.yaml)",
        default="tartan_drive",
        required=True,
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        help="path of the datasets with rosbags",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="../datasets/tartan_drive/",
        type=str,
        help="path for processed dataset (default: ../datasets/tartan_drive/)",
    )
    # number of trajs to process
    parser.add_argument(
        "--num-trajs",
        "-n",
        default=-1,
        type=int,
        help="number of bags to process (default: -1, all)",
    )
    # sampling rate
    parser.add_argument(
        "--sample-rate",
        "-s",
        default=4.0,
        type=float,
        help="sampling rate (default: 4.0 hz)",
    )

    args = parser.parse_args()
    # all caps for the dataset name
    print(f"STARTING PROCESSING {args.dataset_name.upper()} DATASET")
    main(args)
    print(f"FINISHED PROCESSING {args.dataset_name.upper()} DATASET")
