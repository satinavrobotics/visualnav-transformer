import argparse
import os
import time
import subprocess

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import mlflow
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.optim import Adam, AdamW
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from warmup_scheduler import GradualWarmupScheduler

"""
IMPORT YOUR MODEL HERE
"""
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

from visualnav_transformer.train.vint_train.data.vint_dataset import ViNT_Dataset
from visualnav_transformer.train.vint_train.data.feature_dataset import FeatureDataset
from visualnav_transformer.train.vint_train.models.gnm.gnm import GNM
from visualnav_transformer.train.vint_train.models.nomad.nomad import (
    DenseNetwork,
    NoMaD,
)
from visualnav_transformer.train.vint_train.models.sati.sati import Sati
from visualnav_transformer.train.vint_train.models.sati.sati_encoder import SatiEncoder
from visualnav_transformer.train.vint_train.models.nomad.nomad_vint import NoMaD_ViNT
from visualnav_transformer.train.vint_train.models.nomad.utils import replace_bn_with_gn
from visualnav_transformer.train.vint_train.models.vint.vint import ViNT
from visualnav_transformer.train.vint_train.models.vint.vit import ViT
from visualnav_transformer.train.vint_train.training.train_eval_loop import (
    load_model,
    train_eval_loop,
    train_eval_loop_nomad,
)

from visualnav_transformer import ROOT_TRAIN
with open(
        # CHANGE
    # os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r"
    os.path.join(ROOT_TRAIN, "vint_train/data/data_config.yaml"), "r"
) as f:
    data_configs = yaml.safe_load(f)

def log_semaphore_count(label=""):
    try:
        result = subprocess.check_output("ls /dev/shm | grep -c ^sem\\.", shell=True)
        count = int(result.decode().strip())
        print(f"[DEBUG] 🔍 /dev/shm semaphore count {label}: {count}")
        return count
    except Exception as e:
        print(f"[WARN] Could not check /dev/shm semaphores: {e}")
        return -1

def clean_stale_semaphores(threshold=40):
    count = log_semaphore_count("before cleanup")
    if count > threshold:
        print(f"⚠️ Too many semaphores ({count}), cleaning...")
        subprocess.run("ls /dev/shm/sem.* 2>/dev/null | xargs -r rm -v", shell=True)


def main(config):
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True  # good if input sizes don't vary
    transform = [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    transform = transforms.Compose(transform)

    # Load the data
    train_dataset = []
    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    if "clip_goals" not in config:
        config["clip_goals"] = False

    for dataset_index, dataset_name in enumerate(data_configs["datasets"]):
        data_config = data_configs["datasets"][dataset_name]

        if not bool(data_config["available"]):
            continue

        # Skip viz_data dataset - it's only used for visualization, not training
        if dataset_name == "viz_data":
            print(f"Skipping {dataset_name} - visualization dataset only")
            continue
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        train_split = data_config["split"]
        test_split = 1 - train_split

        if train_split != 0.0:
            # Check if using pre-built DINO features
            use_prebuilt_features = config.get("prebuilt_dino", False)

            if not use_prebuilt_features:
                # Use the original ViNT_Dataset
                dataset = ViNT_Dataset(
                    data_folder=data_config["data_folder"],
                    split="train",
                    split_ratio=train_split,
                    dataset_name=dataset_name,
                    dataset_index=dataset_index,
                    image_size=config["image_size"],
                    waypoint_spacing=data_config["waypoint_spacing"],
                    metric_waypoint_spacing=data_config["metric_waypoint_spacing"],
                    min_dist_cat=config["distance"]["min_dist_cat"],
                    max_dist_cat=config["distance"]["max_dist_cat"],
                    min_action_distance=config["action"]["min_dist_cat"],
                    max_action_distance=config["action"]["max_dist_cat"],
                    negative_mining=data_config["negative_mining"],
                    len_traj_pred=config["len_traj_pred"],
                    learn_angle=config["learn_angle"],
                    context_size=config["context_size"],
                    end_slack=data_config["end_slack"],
                    goals_per_obs=data_config["goals_per_obs"],
                    normalize=config["normalize"]
                )
            else:
                # Use the FeatureDataset for pre-built DINO features
                # Construct the feature folder path (assuming it's in a dino_cache_large subfolder)
                feature_folder = os.path.join(data_config["data_folder"], "dino_cache_large")

                dataset = FeatureDataset(
                    data_folder=data_config["data_folder"],
                    feature_folder=feature_folder,
                    split="train",
                    split_ratio=train_split,
                    dataset_name=dataset_name,
                    dataset_index=dataset_index,
                    image_size=config["image_size"],
                    waypoint_spacing=data_config["waypoint_spacing"],
                    metric_waypoint_spacing=data_config["metric_waypoint_spacing"],
                    min_dist_cat=config["distance"]["min_dist_cat"],
                    max_dist_cat=config["distance"]["max_dist_cat"],
                    min_action_distance=config["action"]["min_dist_cat"],
                    max_action_distance=config["action"]["max_dist_cat"],
                    negative_mining=data_config["negative_mining"],
                    len_traj_pred=config["len_traj_pred"],
                    learn_angle=config["learn_angle"],
                    context_size=config["context_size"],
                    end_slack=data_config["end_slack"],
                    goals_per_obs=data_config["goals_per_obs"],
                    normalize=config["normalize"]
                )

            train_dataset.append(dataset)

        if test_split != 0.0:
            # Check if using pre-built DINO features
            use_prebuilt_features = config.get("prebuilt_dino", False)

            if not use_prebuilt_features:
                # Use the original ViNT_Dataset
                dataset = ViNT_Dataset(
                    data_folder=data_config["data_folder"],
                    split="test",
                    split_ratio=test_split,
                    dataset_name=dataset_name,
                    dataset_index=dataset_index,
                    image_size=config["image_size"],
                    waypoint_spacing=data_config["waypoint_spacing"],
                    metric_waypoint_spacing=data_config["metric_waypoint_spacing"],
                    min_dist_cat=config["distance"]["min_dist_cat"],
                    max_dist_cat=config["distance"]["max_dist_cat"],
                    min_action_distance=config["action"]["min_dist_cat"],
                    max_action_distance=config["action"]["max_dist_cat"],
                    negative_mining=data_config["negative_mining"],
                    len_traj_pred=config["len_traj_pred"],
                    learn_angle=config["learn_angle"],
                    context_size=config["context_size"],
                    end_slack=data_config["end_slack"],
                    goals_per_obs=data_config["goals_per_obs"],
                    normalize=config["normalize"]
                )
            else:
                # Use the FeatureDataset for pre-built DINO features
                # Construct the feature folder path (assuming it's in a dino_cache_large subfolder)
                feature_folder = os.path.join(data_config["data_folder"], "dino_cache_large")

                dataset = FeatureDataset(
                    data_folder=data_config["data_folder"],
                    feature_folder=feature_folder,
                    split="test",
                    split_ratio=test_split,
                    dataset_name=dataset_name,
                    dataset_index=dataset_index,
                    image_size=config["image_size"],
                    waypoint_spacing=data_config["waypoint_spacing"],
                    metric_waypoint_spacing=data_config["metric_waypoint_spacing"],
                    min_dist_cat=config["distance"]["min_dist_cat"],
                    max_dist_cat=config["distance"]["max_dist_cat"],
                    min_action_distance=config["action"]["min_dist_cat"],
                    max_action_distance=config["action"]["max_dist_cat"],
                    negative_mining=data_config["negative_mining"],
                    len_traj_pred=config["len_traj_pred"],
                    learn_angle=config["learn_angle"],
                    context_size=config["context_size"],
                    end_slack=data_config["end_slack"],
                    goals_per_obs=data_config["goals_per_obs"],
                    normalize=config["normalize"]
                )

            # Use dataset_name as the dataset_type if not explicitly defined
            dataset_type = dataset_name
            test_dataloaders[dataset_type] = dataset

    # combine all the datasets from different robots
    train_dataset = ConcatDataset(train_dataset)

    # Reduce memory usage for DataLoader to prevent worker crashes
    # If using pre-built DINO features, use minimal workers and disable persistent workers
    use_prebuilt_features = config.get("prebuilt_dino", False)
    if use_prebuilt_features:
        print("Using minimal DataLoader settings for pre-built DINO features to prevent memory issues")
        num_workers = 0  # Use no workers (single-process loading)
        persistent_workers = False  # Disable persistent workers
        batch_size = min(64, config["batch_size"])  # Reduce batch size if needed
        print(f"Using batch size {batch_size} and {num_workers} workers")
    else:
        num_workers = config["num_workers"]
        persistent_workers = True
        batch_size = config["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=persistent_workers,
        pin_memory=False,  # Disable pin_memory to reduce memory usage
    )

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type, dataset in test_dataloaders.items():
        # Use the same minimal settings for test dataloaders when using pre-built DINO features
        if use_prebuilt_features:
            test_batch_size = min(64, config["eval_batch_size"])
            test_num_workers = 0
        else:
            test_batch_size = config["eval_batch_size"]
            test_num_workers = 0  # Already set to 0

        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=test_batch_size,
            shuffle=True,
            num_workers=test_num_workers,
            drop_last=False,
        )

    # Create the model
    if config["model_type"] == "gnm":
        model = GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    elif config["model_type"] == "vint":
        model = ViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    elif config["model_type"] == "nomad" or config["model_type"] == "pogany":
        if config["vision_encoder"] == "nomad_vint" or config["vision_encoder"] == "pogany_encoder":
            # Pass the prebuilt_dino flag to NoMaD_ViNT
            use_prebuilt_features = config.get("prebuilt_dino", False)
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                obs_encoder=config["obs_encoder"],  # Pass the obs_encoder parameter
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
                use_prebuilt_features=use_prebuilt_features,
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vib":
            vision_encoder = ViB(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vit":
            vision_encoder = ViT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                image_size=config["image_size"],
                patch_size=config["patch_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else:
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")

        noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])

        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
    else:
        raise ValueError(f"Model {config['model']} not supported")

    if config["clipping"]:
        print("Clipping gradients to", config["max_norm"])
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * config["max_norm"], config["max_norm"]
                )
            )

    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine":
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.0,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "plateau":
            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")

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
        latest_checkpoint = torch.load(
            latest_path
        )  # f"cuda:{}" if torch.cuda.is_available() else "cpu")
        load_model(model, config["model_type"], latest_checkpoint)
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

    if config["model_type"] == "vint" or config["model_type"] == "gnm":
        train_eval_loop(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            normalized=config["normalize"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            learn_angle=config["learn_angle"],
            alpha=config["alpha"],
            use_mlflow=config["use_mlflow"],
            eval_fraction=config["eval_fraction"],
        )
    elif config["model_type"] == "nomad":
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
    elif config["model_type"] == "pogany":
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

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="config/vint.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    with open("config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)

    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    os.makedirs(
        config[
            "project_folder"
        ],  # should error if dir already exists to avoid overwriting and old project
    )

    if config["use_mlflow"]:
        mlflow.set_tracking_uri("http://localhost:5003")
        mlflow.set_experiment(config["project_name"])
        mlflow.start_run(run_name=config["run_name"])

        # log parameters
        mlflow.log_params({k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))})

        # optionally log the config file
        mlflow.log_artifact(args.config)

    print(config)

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
