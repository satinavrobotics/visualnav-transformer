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

from visualnav_transformer.train.vint_train.data.load_dataset import load_dataset, load_viz_dataset
from visualnav_transformer.train.vint_train.models.load_model import load_model
from visualnav_transformer.train.performance_utils import clean_stale_semaphores, log_semaphore_count, setup_gpu
from visualnav_transformer.train.vint_train.training.train_eval_loop import load_checkpoint, train_eval_loop_nomad
from visualnav_transformer.train.vint_train.logging.cli_formatter import (
    Colors, Symbols, print_header, print_info, print_success, print_warning,
    print_error, TrainingProgressTracker, format_time
)

"""
IMPORT YOUR MODEL HERE
"""

from visualnav_transformer import ROOT_TRAIN
with open(
    os.path.join("/app/visualnav-transformer/config/data/data_config.yaml"), "r"
    #os.path.join(ROOT_TRAIN, "vint_train/data/data_config.yaml"), "r"
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
    use_prebuilt_features = config.get("prebuilt_dino", False)
    if use_prebuilt_features:
        viz_datasets = load_viz_dataset(config, data_configs)
    else:
        viz_datasets = None

    model, noise_scheduler = load_model(config)
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    lr = float(config["lr"])
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    if config["warmup"]:
        print_info("Using warmup scheduler", Symbols.LIGHTNING)
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=config["warmup_epochs"],
            after_scheduler=scheduler,
        )

    current_epoch = 0
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        print_info(f"Loading model from {load_project_folder}", Symbols.LOAD)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path, weights_only=True)
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
        viz_dataloader=viz_datasets,
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
        beta=float(config.get("beta", 1e-4)),
        use_mlflow=config["use_mlflow"],
        eval_fraction=config["eval_fraction"],
        eval_freq=config["eval_freq"],
    )

    print_header("Training Completed Successfully!", Symbols.SUCCESS, Colors.BRIGHT_GREEN)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")
    parser.add_argument(
        "--config",
        "-c",
        default="/app/visualnav-transformer/config/train/prebuilt_dino.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    # Load config
    with open("/app/visualnav-transformer/config/train/defaults.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)

    # Load GoalGMC config if specified
    if "goal_gmc_config_path" in config:
        goal_gmc_config_path = os.path.join("/app/visualnav-transformer", config["goal_gmc_config_path"])
        if os.path.exists(goal_gmc_config_path):
            print_info(f"Loading GoalGMC config from: {goal_gmc_config_path}", Symbols.LOAD)
            with open(goal_gmc_config_path, "r") as f:
                goal_gmc_full_config = yaml.safe_load(f)
            # Extract only the model configuration for GoalGMC
            if "model" in goal_gmc_full_config:
                config["goal_gmc"].update({
                    "name": goal_gmc_full_config["model"].get("model", "goal_gmc"),
                    "common_dim": goal_gmc_full_config["model"].get("common_dim", 64),
                    "latent_dim": goal_gmc_full_config["model"].get("latent_dim", 64),
                    "loss_type": goal_gmc_full_config["model"].get("loss_type", "infonce")
                })
            # Extract training configuration for temperature settings
            if "training" in goal_gmc_full_config:
                config["goal_gmc"].update({
                    "learnable_temperature": goal_gmc_full_config["training"].get("learnable_temperature", False),
                    "initial_temperature": goal_gmc_full_config["training"].get("temperature", 0.1)
                })
            print_success(f"Updated GoalGMC config: {config['goal_gmc']}", Symbols.SUCCESS)
        else:
            print_warning(f"GoalGMC config file not found at {goal_gmc_config_path}, using defaults", Symbols.WARNING)

    # Handle GoalGMC weights path
    if "goal_gmc_weights_path" in config and config["goal_gmc_weights_path"] is not None:
        goal_gmc_weights_path = os.path.join("/app/visualnav-transformer", config["goal_gmc_weights_path"])
        if os.path.exists(goal_gmc_weights_path):
            config["goal_gmc_weights_path"] = goal_gmc_weights_path
            print_info(f"GoalGMC weights will be loaded from: {goal_gmc_weights_path}", Symbols.LOAD)
        else:
            print_warning(f"GoalGMC weights file not found at {goal_gmc_weights_path}, proceeding without pre-trained weights", Symbols.WARNING)
            config["goal_gmc_weights_path"] = None

    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join("logs", config["project_name"], config["run_name"])
    os.makedirs(config["project_folder"])

    # Display training startup banner
    print_header(f"ViNT Training - {config['project_name']}", Symbols.ROCKET, Colors.BRIGHT_CYAN)
    print_info(f"Run name: {config['run_name']}", Symbols.INFO)
    print_info(f"Model type: {config['model_type']}", Symbols.INFO)
    print_info(f"Epochs: {config['epochs']}", Symbols.INFO)
    print_info(f"Learning rate: {config['lr']}", Symbols.INFO)
    print_info(f"Batch size: {config['batch_size']}", Symbols.INFO)
    print_info(f"Project folder: {config['project_folder']}", Symbols.INFO)

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
        print_info("Starting training process...", Symbols.ROCKET)
        training_start_time = time.time()
        main(config)
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        print_success(f"Total training time: {format_time(total_training_time)}", Symbols.CLOCK)
    except Exception as e:
        print_error(f"Training crashed: {e}", Symbols.ERROR)
        log_semaphore_count("on crash")
        raise

    log_semaphore_count("after training")
