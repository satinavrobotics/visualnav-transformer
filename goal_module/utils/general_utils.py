from models.gmc import GoalGMC
from training.dca_evaluation_trainer import DCAEvaluator
from data_modules.class_dataset import *


def setup_model(scenario, model, model_config, scenario_config, data_module=None, train_config=None):
    if model == "gmc":
        if scenario == "pogany" or scenario == "dummy_goal":
            # Get temperature settings from train_config if available
            learnable_temperature = False
            initial_temperature = 0.1
            if train_config is not None:
                learnable_temperature = train_config.get("learnable_temperature", False)
                initial_temperature = train_config.get("temperature", 0.1)

            return GoalGMC(
                name=model_config["model"],
                common_dim=model_config["common_dim"],
                latent_dim=model_config["latent_dim"],
                loss_type=model_config["loss_type"],
                learnable_temperature=learnable_temperature,
                initial_temperature=initial_temperature,
            )
        else:
            raise ValueError(
                "[Model Setup] Selected scenario not yet implemented for GMC model: "
                + str(scenario)
            )

    else:
        raise ValueError(
            "[Model Setup] Selected model not yet implemented: " + str(model)
        )


def setup_data_module(scenario, experiment_config, scenario_config, train_config, data_config):
    if experiment_config["stage"] == "evaluate_dca":
        return DCADataModule(
            dataset=scenario,
            data_dir=scenario_config["data_dir"],
            train_config=train_config,
            data_config=data_config
        )
    else:
        return ClassificationDataModule(
            dataset=scenario,
            data_dir=scenario_config["data_dir"],
            train_config=train_config,
            data_config=data_config,
        )


def setup_dca_evaluation_trainer(model, machine_path, scenario, config):
    return DCAEvaluator(
        model=model,
        scenario=scenario,
        machine_path=machine_path,
        minimum_cluster_size=config["minimum_cluster_size"],
        unique_modality_idxs=config["unique_modality_idxs"],
        unique_modality_dims=config["unique_modality_dims"],
        partial_modalities_idxs=config["partial_modalities_idxs"],
    )



"""

Loading functions

"""



def load_model(config, model_file):

    model = setup_model(
        scenario=config["experiment"]["scenario"],
        model=config["experiment"]["model"],
        model_config=config["model"],
        scenario_config=config["scenario"],
        train_config=config.get("training", {}),
    )

    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint["state_dict"])

    # Freeze model
    model.freeze()

    return model



"""


General functions

"""


def flatten_dict(dd, separator="_", prefix=""):
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )

