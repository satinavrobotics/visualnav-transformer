import argparse
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import mlflow
import yaml

from torch.optim import AdamW
from torchvision import transforms
from warmup_scheduler import GradualWarmupScheduler

from visualnav_transformer.train.vint_train.data.load_dataset import load_dataset
from visualnav_transformer.train.vint_train.models.load_model import load_model
from visualnav_transformer.train.performance_utils import clean_stale_semaphores, log_semaphore_count, setup_gpu
from visualnav_transformer.train.vint_train.training.train_eval_loop import load_checkpoint, train_eval_loop_nomad

"""
IMPORT YOUR MODEL HERE
"""

from visualnav_transformer import ROOT_TRAIN
with open(
    os.path.join(ROOT_TRAIN, "vint_train/data/data_config.yaml"), "r"
) as f:
    data_configs = yaml.safe_load(f)


def main(config):
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    device = setup_gpu(config)

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True
    cudnn.benchmark = True  # good if input sizes don't vary
    
    train_loader, test_dataloaders = load_dataset(config, data_configs)
    model, noise_scheduler = load_model(config)
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    lr = float(config["lr"])
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    if config["warmup"]:
        print("Using warmup scheduler")
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=config["warmup_epochs"],
            after_scheduler=scheduler,
        )

    current_epoch = 0
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path)
        load_checkpoint(model, config["model_type"], latest_checkpoint)
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1
            
    # Multi-GPU
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)

    if "load_run" in config:  # load optimizer and scheduler after data parallel
        if "optimizer" in latest_checkpoint:
            optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
        if scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())

    train_eval_loop_nomad(
        train_model=config["train"],
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        noise_scheduler=noise_scheduler,
        train_loader=train_loader,
        test_dataloaders=test_dataloaders,
        transform=transform,
        goal_mask_prob=config["goal_mask_prob"],
        epochs=config["epochs"],
        device=device,
        project_folder=config["project_folder"],
        print_log_freq=config["print_log_freq"],
        mlflow_log_freq=config["mlflow_log_freq"],
        image_log_freq=config["image_log_freq"],
        num_images_log=config["num_images_log"],
        current_epoch=current_epoch,
        alpha=float(config["alpha"]),
        use_mlflow=config["use_mlflow"],
        eval_fraction=config["eval_fraction"],
        eval_freq=config["eval_freq"],
    )

    print("FINISHED TRAINING")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")
    parser.add_argument(
        "--config",
        "-c",
        default="config/vint.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    # Load config
    with open("config/defaults.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)
    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join("logs", config["project_name"], config["run_name"])
    os.makedirs(config["project_folder"])
    print(config)

    # Log to mlflow
    if config["use_mlflow"]:
        mlflow.set_tracking_uri("http://localhost:5003")
        mlflow.set_experiment(config["project_name"])
        mlflow.start_run(run_name=config["run_name"])
        mlflow.log_params({k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))})
        mlflow.log_artifact(args.config)

    # Added semaphore logging for cpu ram issues
    clean_stale_semaphores()
    log_semaphore_count("after cleanup")

    try:
        main(config)
    except Exception as e:
        print(f"[ERROR] Training crashed: {e}")
        log_semaphore_count("on crash")
        raise

    log_semaphore_count("after training")
