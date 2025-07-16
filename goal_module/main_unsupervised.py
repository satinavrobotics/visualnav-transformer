import os
import torch
import yaml
import argparse
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from training.mlflow_logger import MLFlowLogger
from training.model_trainer import ModelLearner
from training.callbacks import OnEndModelTraining
from utils.general_utils import (
    setup_dca_evaluation_trainer,
    setup_model,
    setup_data_module,
    load_model
)
from pytorch_lightning.callbacks import TQDMProgressBar

AVAIL_GPUS = min(1, torch.cuda.device_count())
torch.set_float32_matmul_precision('medium') # | 'high'


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def log_dir_path(folder, config):
    """Generate log directory path"""
    model_type = str(config["experiment"]["model"])
    exp_name = str(config["experiment"]["scenario"])

    return os.path.join(
        config["machine"]["m_path"],
        "evaluation/",
        model_type + "_" + exp_name,
        f'log_{config["experiment"]["seed"]}',
        folder,
    )


def trained_model_dir_path(file, config):
    """Generate trained model directory path"""
    return os.path.join(
        config["machine"]["m_path"],
        "trained_models/",
        file
    )


def load_hyperparameters(config):
    """Load hyperparameters from config"""
    exp_cfg = config["experiment"]
    scenario_cfg = config["dataset"]
    model_cfg = config["model"]

    return exp_cfg, scenario_cfg, model_cfg


def train_model(config, data_config):
    """Train the model using MLflow for experiment tracking"""

    # Init model input parameters
    exp_cfg, scenario_cfg, model_cfg = load_hyperparameters(config)
    model_train_cfg = config["training"]

    # Set the seeds
    seed_everything(exp_cfg["seed"], workers=True)

    # Init model
    model = setup_model(
        model=exp_cfg["model"],
        scenario=exp_cfg["scenario"],
        scenario_config=scenario_cfg,
        model_config=model_cfg,
        train_config=model_train_cfg)

    # Init Data Module
    data_module = setup_data_module(
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=model_train_cfg,
        data_config=data_config
    )

    # Init Trainer
    model_trainer = ModelLearner(
        model=model,
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=model_train_cfg)

    # MLflow Logger
    mlflow_config = config.get("mlflow", {})
    mlflow_logger = MLFlowLogger(
        experiment_name=mlflow_config.get("experiment_name", "GMC_unsupervised_experiments"),
        tracking_uri=mlflow_config.get("tracking_uri", "file:./mlruns")
    )

    # Log hyperparameters
    mlflow_logger.log_hyperparams({
        "model": exp_cfg["model"],
        "scenario": exp_cfg["scenario"],
        "seed": exp_cfg["seed"],
        "learning_rate": model_train_cfg["learning_rate"],
        "batch_size": model_train_cfg["batch_size"],
        "epochs": model_train_cfg["epochs"],
        "common_dim": model_cfg["common_dim"],
        "latent_dim": model_cfg["latent_dim"],
        "loss_type": model_cfg["loss_type"]
    })

    # Train
    checkpoint_dir = log_dir_path("checkpoints", config)
    run_id = mlflow_logger.run_id

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{exp_cfg['model']}_{exp_cfg['scenario']}_{run_id}-"
        + "{epoch:02d}",
        monitor="val_loss",
        every_n_epochs=model_train_cfg["snapshot"],
        save_top_k=-1,
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = (
        f"{exp_cfg['model']}_{exp_cfg['scenario']}_{run_id}_last"
    )
    checkpoint_callback.FILE_EXTENSION = ".pth"
    
    # Trainer
    trainer = Trainer(
        accelerator="gpu" if exp_cfg["cuda"] else "cpu",
        devices=AVAIL_GPUS if exp_cfg["cuda"] else 1,
        max_epochs=model_train_cfg["epochs"],
        default_root_dir=log_dir_path("saved_models", config),
        logger=mlflow_logger,
        callbacks=[
            checkpoint_callback, 
            OnEndModelTraining(), 
            TQDMProgressBar(refresh_rate=20)
        ])

    # Train
    trainer.fit(model_trainer, data_module)

def dca_eval_model(config):
    """Evaluate model using DCA with MLflow tracking"""
    # Init model input parameters
    exp_cfg, scenario_cfg, _ = load_hyperparameters(config)
    dca_eval_cfg = config["dca_evaluation"]

    # Set the seeds
    seed_everything(dca_eval_cfg["random_seed"], workers=True)

    # Load model
    model_file = trained_model_dir_path(exp_cfg['model'] + "_" + exp_cfg['scenario'] + "_model.pth.tar", config)
    model = load_model(config=config, model_file=model_file)

    # Init Trainer
    dca_trainer = setup_dca_evaluation_trainer(
        model=model,
        machine_path=config["machine"]["m_path"],
        scenario=exp_cfg["scenario"],
        config=dca_eval_cfg,
    )

    # Init Data Module
    dca_data_module = setup_data_module(
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=dca_eval_cfg,
    )

    # MLflow Logger
    mlflow_config = config.get("mlflow", {})
    mlflow_logger = MLFlowLogger(
        experiment_name=mlflow_config.get("experiment_name", "goal_module"),
        tracking_uri=mlflow_config.get("tracking_uri", "http://localhost:5003")
    )

    # Trainer
    trainer = Trainer(
        accelerator="gpu" if AVAIL_GPUS > 0 else "cpu",
        devices=AVAIL_GPUS if AVAIL_GPUS > 0 else "auto",
        default_root_dir=log_dir_path("results_dca_evaluation", config),
        logger=mlflow_logger,
        enable_progress_bar=True,
    )

    trainer.test(dca_trainer, dca_data_module)
    return



def main():
    """Main function to run experiments with YAML config"""
    config_default = "/app/visualnav-transformer/config/goal_module/config.yaml"
    data_config_default = "/app/visualnav-transformer/config/data/data_config_goal_module.yaml"
    
    parser = argparse.ArgumentParser(description='GMC Unsupervised Experiments')
    parser.add_argument('--config', type=str, default=config_default,
                        help='Path to configuration file')
    parser.add_argument('--data_config', type=str, default=data_config_default,
                        help='Path to data configuration file')

    args = parser.parse_args()

    # Load single configuration file
    config = load_config(args.config)
    data_config = load_config(args.data_config)

    # Run experiment
    if config["experiment"]["stage"] == "train_model":
        os.makedirs(log_dir_path("saved_models", config), exist_ok=True)
        os.makedirs(log_dir_path("checkpoints", config), exist_ok=True)
        train_model(config, data_config)

    elif config["experiment"]["stage"] == "evaluate_dca":
        os.makedirs(log_dir_path("results_dca_evaluation", config), exist_ok=True)
        dca_eval_model(config)

    else:
        raise ValueError(
            "[Unsupervised Experiment] Incorrect stage of pipeline selected: " + str(config["experiment"]["stage"])
        )


if __name__ == "__main__":
    main()

