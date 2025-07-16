import torch
import math
from pytorch_lightning import LightningModule
from .gmc_networks import *


class GMC(LightningModule):
    def __init__(self, name, common_dim, latent_dim, learnable_temperature=False, initial_temperature=0.1):
        super(GMC, self).__init__()

        self.name = name
        self.common_dim = common_dim
        self.latent_dim = latent_dim
        self.learnable_temperature = learnable_temperature

        # Initialize temperature parameter
        if learnable_temperature:
            # Use logit_scale parameterization for learnable temperature
            self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(1/0.07)))
        else:
            self.register_buffer('temperature', torch.tensor(initial_temperature))

        self.image_processor = None
        self.label_processor = None
        self.joint_processor = None
        self.processors = [
            self.image_processor,
            self.label_processor,
            self.joint_processor,
        ]

        self.encoder = None

    @property
    def temperature(self):
        """Compute temperature from logit_scale when learnable, otherwise return fixed temperature."""
        if self.learnable_temperature:
            # Clamp logit_scale to prevent extreme values
            self.logit_scale.data.clamp_(0, 4.6052)
            return 1.0 / self.logit_scale.exp()
        else:
            return self._buffers['temperature']

    def encode(self, x, sample=False):

        # If we have complete observations
        if None not in x:
            return self.encoder(self.processors[-1](x))
        else:
            latent_representations = []
            for id_mod in range(len(x)):
                if x[id_mod] is not None:
                    latent_representations.append(self.encoder(self.processors[id_mod](x[id_mod])))

            # Take the average of the latent representations
            latent = torch.stack(latent_representations, dim=0).mean(0)
            return latent

    def forward(self, x):
        # Forward pass through the modality specific encoders
        batch_representations = []
        for processor_idx in range(len(self.processors) - 1):
            mod_representations = self.encoder(
                self.processors[processor_idx](x[processor_idx])
            )
            batch_representations.append(mod_representations)

        # Forward pass through the joint encoder
        joint_representation = self.encoder(self.processors[-1](x))
        batch_representations.append(joint_representation)
        return batch_representations

    def infonce(self, batch_representations, temperature, batch_size):
        joint_mod_loss_sum = 0
        tqdm_dict = {}
        all_avg_pos = []
        for mod in range(len(batch_representations) - 1):
            # Negative pairs: everything that is not in the current joint-modality pair
            out_joint_mod = torch.cat(
                [batch_representations[-1], batch_representations[mod]], dim=0
            )
            # [2*B, 2*B]
            sim_matrix_joint_mod = torch.exp(
                torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / temperature
            )
            # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
            mask_joint_mod = (
                torch.ones_like(sim_matrix_joint_mod)
                - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
            ).bool()
            # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
            sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(
                mask_joint_mod
            ).view(2 * batch_size, -1)

            # Positive pairs: cosine loss joint-modality
            pos_sim_joint_mod = torch.exp(
                torch.sum(
                    batch_representations[-1] * batch_representations[mod], dim=-1
                )
                / temperature
            )
            # [2*B]
            pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1)
            )
            joint_mod_loss_sum += loss_joint_mod
            tqdm_dict[self.processors[mod].loss_name] = loss_joint_mod.mean()
            
            avg_p = self.avg_pos_probability(sim_matrix_joint_mod, pos_sim_joint_mod)
            all_avg_pos.append(avg_p)
            tqdm_dict[self.processors[mod].loss_name + '_avg_pos_prob'] = avg_p

        loss = torch.mean(joint_mod_loss_sum)
        tqdm_dict["loss"] = loss
        tqdm_dict["avg_pos_prob"] = sum(all_avg_pos) / len(all_avg_pos)
        tqdm_dict["temperature"] = temperature
        return loss, tqdm_dict

    def training_step(self, data, train_params):

        # Use model's temperature parameter if learnable, otherwise use config value
        if self.learnable_temperature:
            temperature = self.temperature
        else:
            temperature = train_params["temperature"]

        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        batch_representations = self.forward(data)

        # Compute contrastive loss
        loss, tqdm_dict = self.infonce(batch_representations, temperature, batch_size)
        return loss, tqdm_dict

    def validation_step(self, data, train_params):

        # Use model's temperature parameter if learnable, otherwise use config value
        if self.learnable_temperature:
            temperature = self.temperature
        else:
            temperature = train_params["temperature"]

        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        batch_representations = self.forward(data)
        # Compute contrastive loss
        loss, tqdm_dict = self.infonce(batch_representations, temperature, batch_size)
        return tqdm_dict
    
    def avg_pos_probability(self, sim_matrix: torch.Tensor, pos_sim: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sim_matrix: Tensor of shape [2B, 2B-1], 
                        containing exp(sim/temperature) for all non‐diagonal pairs.
            pos_sim:    Tensor of shape [2B], 
                        containing exp(sim_pos/temperature) for each positive pair (duplicated).
        Returns:
            avg_p:      scalar Tensor, the arithmetic mean of p_i = pos_sim[i] / sim_matrix.sum(dim=-1)[i]
        """
        # sum over each row’s negatives
        neg_sums = sim_matrix.sum(dim=-1)         # [2B]
        # probability mass on the positive for each i
        pos_probs = pos_sim / (pos_sim + neg_sums) # [2B]
        # average over the batch
        avg_p = pos_probs.mean()
        return avg_p


class GoalGMC(GMC):
    def __init__(self, name, common_dim, latent_dim, loss_type, learnable_temperature=False, initial_temperature=0.1,
                 memory_efficient=False, goal_encoder_type=None):
        super(GoalGMC, self).__init__(name, common_dim, latent_dim, learnable_temperature, initial_temperature)

        self.memory_efficient = memory_efficient
        self.goal_encoder_type = goal_encoder_type

        # Always create all processors initially
        self.features_processor = ImageFeaturePairProcessor(common_dim=common_dim)
        self.coord_processor = CoordProcessor(common_dim=common_dim)
        self.joint_processor = JointProcessor(common_dim=common_dim)
        self.processors = [
            self.features_processor,
            self.coord_processor,
            self.joint_processor,
        ]
        self.loss_type = loss_type

        self.encoder = SharedProjection(common_dim=common_dim, latent_dim=latent_dim)

        # Apply memory optimization if requested
        if memory_efficient and goal_encoder_type is not None:
            self._optimize_memory_usage(goal_encoder_type)
        
    def _optimize_memory_usage(self, goal_encoder_type):
        """
        Remove unused processors from memory based on goal encoder type.

        Args:
            goal_encoder_type: Type of goal encoder being used
                - "image_pair" or "feature_pair": only needs features_processor
                - "position": only needs coord_processor
                - "image_pair_position" or "feature_pair_position": needs joint_processor
        """
        if goal_encoder_type in ["image_pair", "feature_pair"]:
            # Only keep features processor
            del self.coord_processor
            del self.joint_processor
            self.coord_processor = None
            self.joint_processor = None
            self.processors = [self.features_processor, None, None]

        elif goal_encoder_type == "position":
            # Only keep coord processor
            del self.features_processor
            del self.joint_processor
            self.features_processor = None
            self.joint_processor = None
            self.processors = [None, self.coord_processor, None]

        elif goal_encoder_type in ["image_pair_position", "feature_pair_position"]:
            # Only keep joint processor (which internally uses both features and coord)
            del self.features_processor
            del self.coord_processor
            self.features_processor = None
            self.coord_processor = None
            self.processors = [None, None, self.joint_processor]

        # Force garbage collection to free memory
        import gc
        gc.collect()

    def load_weights_from_checkpoint(self, checkpoint_path):
        """
        Load weights from a trained GoalGMC checkpoint file.

        Args:
            checkpoint_path: Path to the .pth.tar checkpoint file
        """
        import torch
        import os

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract state dict - handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            # Assume the checkpoint is the state dict itself
            state_dict = checkpoint

        # Load the state dict
        self.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded GoalGMC weights from: {checkpoint_path}")

    def encode_forward(self, encoder_type, **kwargs):
        if encoder_type == "features":
            if self.features_processor is None:
                raise RuntimeError("Features processor not available - check memory optimization settings")
            return self.encoder(self.features_processor(**kwargs))
        elif encoder_type == "coord":
            if self.coord_processor is None:
                raise RuntimeError("Coord processor not available - check memory optimization settings")
            return self.encoder(self.coord_processor(**kwargs))
        elif encoder_type == "joint":
            if self.joint_processor is None:
                raise RuntimeError("Joint processor not available - check memory optimization settings")
            return self.encoder(self.joint_processor(**kwargs))
        else:
            raise ValueError(f"Invalid encoder type: {encoder_type}")