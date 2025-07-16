import torch.optim as optim
from pytorch_lightning import LightningModule


class ModelLearner(LightningModule):
    def __init__(
        self, model, scenario, train_config, scenario_config, experiment_config,
    ):
        super(ModelLearner, self).__init__()

        self.model = model
        self.scenario = scenario
        self.experiment_config = experiment_config
        self.train_config = train_config
        self.scenario_config = scenario_config

    def configure_optimizers(self):
        optimiser = optim.Adam(
            self.model.parameters(), lr=self.train_config["learning_rate"]
        )
        return optimiser

    def training_step(self, batch, batch_idx):
        # Forward pass through the encoders
        if self.scenario == 'pogany' or self.scenario == 'dummy_goal':
            # For dummy_goal: (features, goal_features), goal_position
            data = [batch[0], batch[1]]
        else:
            raise ValueError(
                "[Model Learner] Scenario not yet implemented: " + str(self.scenario)
            )
        loss, tqdm_dict = self.model.training_step(data, self.train_config)

        # Log metrics using the new PyTorch Lightning 2.0 approach
        for key, value in tqdm_dict.items():
            self.log(f"train_{key}", value, on_step=True, on_epoch=True, prog_bar=True)
            # Also log to MLflow
            if hasattr(self.logger, 'log_metric'):
                self.logger.log_metric(f"train_{key}", value, self.current_epoch)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.scenario == 'pogany' or self.scenario == 'dummy_goal':
            data = [batch[0], batch[1]]
        else:
            raise ValueError(
                "[Model Learner] Scenario not yet implemented: " + str(self.scenario)
            )

        output_dict = self.model.validation_step(data, self.train_config)

        # Log validation metrics using the new PyTorch Lightning 2.0 approach
        for key, value in output_dict.items():
            self.log(f"val_{key}", value, on_step=False, on_epoch=True, prog_bar=True)
            # Also log to MLflow
            if hasattr(self.logger, 'log_metric'):
                self.logger.log_metric(f"val_{key}", value, self.current_epoch)

        return output_dict


