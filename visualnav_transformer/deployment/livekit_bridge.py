import asyncio
import logging
import os
import time
from signal import SIGINT, SIGTERM
import numpy as np
import threading
from livekit import api, rtc

import rclpy
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from rclpy.node import Node

# ROS
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

# UTILS
from visualnav_transformer.deployment.topic_names import (
    IMAGE_TOPIC,
    SAMPLED_ACTIONS_TOPIC,
    WAYPOINT_TOPIC,
)
from visualnav_transformer.deployment.utils import (
    load_model,
    msg_to_pil,
    to_numpy,
    transform_images,
)
from visualnav_transformer.train.vint_train.training.train_utils import get_action

# CONSTANTS
from visualnav_transformer import ROOT
MODEL_WEIGHTS_PATH = os.path.join(ROOT, "model_weights")
ROBOT_CONFIG_PATH = os.path.join(ROOT, "config/robot.yaml")
MODEL_CONFIG_PATH = os.path.join(ROOT, "config/models.yaml")
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]

# GLOBALS
context_queue = []
context_size = None

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ensure LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET are set

tasks = set()


async def main(room: rtc.Room) -> None:
    video_stream = None

    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, *_):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            nonlocal video_stream
            if video_stream is not None:
                # only process the first stream received
                return

            print("subscribed to track: " + track.name)
            video_stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGB24)
            task = asyncio.create_task(frame_loop(video_stream))
            tasks.add(task)
            task.add_done_callback(tasks.remove)

    token = (
        api.AccessToken()
        .with_identity("python-bot")
        .with_name("Python Bot")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room="admin@satinavrobotics.com",
            )
        )
    )
    await room.connect(os.getenv("LIVEKIT_URL"), token.to_jwt())
    print("connected to room: " + room.name)




async def frame_loop(video_stream: rtc.VideoStream) -> None:
    use_every_n_frame = 5
    frame_count = 0
    waypoint: int = 2,
    num_samples: int = 8,
    model, model_params, noise_scheduler, num_diffusion_iters, = init_models()
    thread = threading.Thread(target=model_inference, daemon=True)
    thread.start()
    
    # initialize model
    async for frame_event in video_stream:
        frame_count += 1
        if frame_count % use_every_n_frame != 0:
            continue
        
        buffer = frame_event.frame
        # rgb image
        arr = np.frombuffer(buffer.data, dtype=np.uint8)
        arr = arr.reshape((buffer.height, buffer.width, 3))
        callback_obs(arr)
        
    # close model 
    
def callback_obs(img):
    obs_img = PILImage.fromarray(img)
    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)
        

def init_models(
    args_model: str = "nomad",
):
    global context_size

    # load model parameters
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = os.path.join(ROOT, model_paths[args_model]["config_path"])
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    context_size = model_params["context_size"]

    # load model weights
    ckpth_path = os.path.join(ROOT, model_paths[args_model]["ckpt_path"])
    if os.path.exists(ckpth_path):
        print(f"Loading model from {ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    model = load_model(
        ckpth_path,
        model_params,
        device,
    )
    model = model.to(device)
    model.eval()

    num_diffusion_iters = model_params["num_diffusion_iters"]
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=model_params["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    print("Starting exploration")
    return model, model_params, noise_scheduler, num_diffusion_iters, args_waypoint, args_num_samples



def model_inference():
    while True:
        loop_start_time = time.time()
        # EXPLORATION MODE
        waypoint_msg = Float32MultiArray()
        if len(context_queue) > model_params["context_size"]:
            obs_images = transform_images(
                context_queue, model_params["image_size"], center_crop=False
            )
            obs_images = obs_images.to(device)
            fake_goal = torch.randn((1, 3, *model_params["image_size"])).to(device)
            mask = torch.ones(1).long().to(device)  # ignore the goal

            # infer action
            with torch.no_grad():
                # encoder vision features
                obs_cond = model(
                    "vision_encoder",
                    obs_img=obs_images,
                    goal_img=fake_goal,
                    input_goal_mask=mask,
                )

                # (B, obs_horizon * obs_dim)
                if len(obs_cond.shape) == 2:
                    obs_cond = obs_cond.repeat(args_num_samples, 1)
                else:
                    obs_cond = obs_cond.repeat(args_num_samples, 1, 1)

                # initialize action from Gaussian noise
                noisy_action = torch.randn(
                    (args_num_samples, model_params["len_traj_pred"], 2), device=device
                )
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

                start_time = time.time()
                for k in noise_scheduler.timesteps[:]:
                    # predict noise
                    noise_pred = model(
                        "noise_pred_net",
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond,
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred, timestep=k, sample=naction
                    ).prev_sample
                print("time elapsed:", time.time() - start_time)

            naction = to_numpy(get_action(naction))
            sampled_actions_msg = Float32MultiArray()
            sampled_actions_msg.data = np.concatenate(
                (np.array([0]), naction.flatten())
            ).tolist()
            node.sampled_actions_pub.publish(sampled_actions_msg)

            #print(naction)
            naction = naction[0]  # change this based on heuristic

            #print(args_waypoint)
            chosen_waypoint = naction[args_waypoint]

            if model_params["normalize"]:
                chosen_waypoint *= MAX_V / RATE
            waypoint_msg.data = chosen_waypoint.tolist()
            node.waypoint_pub.publish(waypoint_msg)

            print("Published waypoint")
            elapsed_time = time.time() - loop_start_time
            sleep_time = max(0, (1.0 / RATE) - elapsed_time)
            time.sleep(sleep_time)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler("nomad.log"), logging.StreamHandler()],
    )

    loop = asyncio.get_event_loop()
    room = rtc.Room(loop=loop)

    async def cleanup():
        await room.disconnect()
        loop.stop()

    asyncio.ensure_future(main(room))
    for signal in [SIGINT, SIGTERM]:
        loop.add_signal_handler(signal, lambda: asyncio.ensure_future(cleanup()))

    try:
        loop.run_forever()
    finally:
        loop.close()