import os
import torch
from pytorch_lightning.callbacks import Callback


class OnEndModelTraining(Callback):
    def on_init_end(self, trainer):
        print(f"Initialised Model Trainer with {trainer.default_root_dir}")

    def on_train_end(self, trainer, pl_module):

        torch.save(
            {"state_dict": pl_module.model.state_dict()},
            os.path.join(
                trainer.default_root_dir, f"{pl_module.model.name}_dummy_goal_model.pth.tar"
            ),
        )

        print(
            f"Model {pl_module.model.name} trained for {trainer.max_epochs} epochs in the Dummy Goal dataset saved to {trainer.default_root_dir}"
        )
