import os
import yaml
import subprocess

def get_rosbag_topics(bag_path):
    """Extracts the list of topics from a rosbag."""
    try:
        result = subprocess.run(["rosbag", "info", "--yaml", bag_path], capture_output=True, text=True, check=True)
        bag_info = yaml.safe_load(result.stdout)
        return bag_info.get("topics", [])
    except subprocess.CalledProcessError as e:
        print(f"Error processing {bag_path}: {e}")
        return []

def generate_yaml_config(bag_folder, output_yaml):
    """Scans all rosbags in the folder and generates a YAML config file."""
    rosbag_files = [f for f in os.listdir(bag_folder) if f.endswith(".bag")]
    
    odom_topics = set()
    image_topics = set()
    
    for bag_file in rosbag_files:
        bag_path = os.path.join(bag_folder, bag_file)
        topics = get_rosbag_topics(bag_path)
        
        for topic_info in topics:
            topic_name = topic_info.get("topic", "")
            if "odom" in topic_name.lower():
                odom_topics.add(topic_name)
            if "image" in topic_name.lower():
                image_topics.add(topic_name)
    
    config = {
        "odomtopics": list(odom_topics),
        "imtopics": list(image_topics),
        "ang_offset": 0.0,
        "img_process_func": "process_scand_img",
        "odom_process_func": "nav_to_xy_yaw"
    }
    
    with open(output_yaml, "w") as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)
    
    print(f"YAML configuration saved to {output_yaml}")

# Example usage:
generate_yaml_config("Sati_data/SCAND_new/scand_4_3", "Sati_data/SCAND_new/scand_config.yaml")
