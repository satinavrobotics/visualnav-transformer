from typing import Optional, Tuple

import torch
import torch.nn as nn
import math

from visualnav_transformer.train.vint_train.models.ft_extractor import (
    EfficientNetExtractor, DiNOV2Extractor
)
from visualnav_transformer.train.vint_train.models.utils import PositionalEncoding
from goal_module.models.gmc import GoalGMC


class NoMaD_ViNT(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        obs_encoder: Optional[str] = "dinov2-small",
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
        use_prebuilt_features: bool = False,
        goal_encoder_type: str = "goal_image",  # New parameter
        goal_gmc_config: Optional[dict] = None,  # GoalGMC configuration
        goal_gmc_weights_path: Optional[str] = None,  # Path to trained GoalGMC weights
    ) -> None:
        """
        NoMaD ViNT Encoder class

        Args:
            context_size: Number of context frames
            obs_encoder: Type of observation encoder
            obs_encoding_size: Size of observation encoding
            mha_num_attention_heads: Number of attention heads
            mha_num_attention_layers: Number of attention layers
            mha_ff_dim_factor: Feedforward dimension factor
            use_prebuilt_features: Whether to use pre-built features
            goal_encoder_type: Type of goal encoder (image_pair, position, image_pair_position, feature_pair, feature_pair_position)
            goal_gmc_config: Configuration dictionary for GoalGMC module
            goal_gmc_weights_path: Path to trained GoalGMC weights file (.pth.tar)
        """
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size
        self.use_prebuilt_features = use_prebuilt_features
        self.goal_encoder_type = goal_encoder_type
        self.goal_gmc_weights_path = goal_gmc_weights_path

        # Set default GoalGMC config if not provided
        if goal_gmc_config is None:
            goal_gmc_config = {
                "name": "goal_gmc",
                "common_dim": 64,
                "latent_dim": 64,
                "loss_type": "infonce",
                "learnable_temperature": False,
                "initial_temperature": 0.1
            }
        self.goal_gmc_config = goal_gmc_config

        # Validate goal encoder type
        valid_encoder_types_not_prebuilt = ["image_pair", "position", "image_pair_position"]
        valid_encoder_types_prebuilt = ["feature_pair", "position", "feature_pair_position"]
        
        if not use_prebuilt_features:
            if goal_encoder_type not in valid_encoder_types_not_prebuilt:
                raise ValueError(f"goal_encoder_type must be one of {valid_encoder_types_not_prebuilt}, got {goal_encoder_type}")

            print(f"Using feature extractor: {obs_encoder.split('-')[0]} with goal encoder type: {goal_encoder_type}")

            if obs_encoder.split("-")[0] == "efficientnet":
                self.obs_encoder = EfficientNetExtractor(obs_encoder, in_channels=3)
                if goal_encoder_type in ["image_pair", "image_pair_position"]:
                    self.goal_encoder = EfficientNetExtractor(obs_encoder, in_channels=3)
            elif obs_encoder.split("-")[0] == "dinov2":
                self.obs_encoder = DiNOV2Extractor(obs_encoder)
                if goal_encoder_type in ["image_pair", "image_pair_position"]:
                    self.goal_encoder = DiNOV2Extractor(obs_encoder)

            self.num_obs_features = self.obs_encoder.num_out_features
        else:
            if goal_encoder_type not in valid_encoder_types_prebuilt:
                raise ValueError(f"goal_encoder_type must be one of {valid_encoder_types_prebuilt}, got {goal_encoder_type}")
            if "large" in obs_encoder:
                self.num_obs_features = 1024
            elif "base" in obs_encoder:
                self.num_obs_features = 768
            elif "small" in obs_encoder:
                self.num_obs_features = 384
            else:
                self.num_obs_features = 384
                
        self.num_goal_features = self.goal_encoding_size

        # Initialize GoalGMC with configuration parameters and memory optimization
        self.goal_gmc = GoalGMC(
            name=self.goal_gmc_config["name"],
            common_dim=self.goal_gmc_config["common_dim"],
            latent_dim=self.goal_gmc_config["latent_dim"],
            loss_type=self.goal_gmc_config["loss_type"],
            learnable_temperature=self.goal_gmc_config["learnable_temperature"],
            initial_temperature=self.goal_gmc_config["initial_temperature"],
            memory_efficient=True,  # Enable memory optimization during training
            goal_encoder_type=self.goal_encoder_type
        )

        # Load pre-trained weights if provided
        if self.goal_gmc_weights_path is not None:
            self.goal_gmc.load_weights_from_checkpoint(self.goal_gmc_weights_path)

        self.num_goal_features = self.goal_gmc.common_dim

        # Initialize compression layers if necessary
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(
                self.num_obs_features, self.obs_encoding_size
            )
        else:
            self.compress_obs_enc = nn.Identity()

        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(
                self.num_goal_features, self.goal_encoding_size
            )
        else:
            self.compress_goal_enc = nn.Identity()

        # Initialize positional encoding and self-attention layers
        self.positional_encoding = PositionalEncoding(
            self.obs_encoding_size, max_seq_len=self.context_size + 2
        )
        self.sa_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.obs_encoding_size,
                nhead=mha_num_attention_heads,
                dim_feedforward=mha_ff_dim_factor * self.obs_encoding_size,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ), 
            num_layers=mha_num_attention_layers
        )

        # Definition of the goal mask (convention: 0 = no mask, 1 = mask)
        self.goal_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)
        self.goal_mask[:, -1] = True  # Mask out the goal
        self.no_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)
        self.all_masks = torch.cat([self.no_mask, self.goal_mask], dim=0)
        self.avg_pool_mask = torch.cat(
            [
                1 - self.no_mask.float(),
                (1 - self.goal_mask.float())
                * ((self.context_size + 2) / (self.context_size + 1)),
            ],
            dim=0,
        )

    def forward(
        self,
        obs_img: torch.tensor, # Either images [128, 12, 240, 320] or pre-computed features [batch_size, context_size+1, feature_dim]
        goal_img: torch.tensor = None, # Either images [128, 3, 240, 320] or pre-computed features [batch_size, feature_dim]
        goal_pos: torch.tensor = None, # Goal positions [batch_size, 2] for position-based encoders
        input_goal_mask: torch.tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = obs_img.device

        if not self.use_prebuilt_features:
            
            # Get the observation encoding
            obs_img = torch.split(obs_img, 3, dim=1) # obs_img shape: 4
            obs_img = torch.concat(obs_img, dim=0) # obs_img shape: 512
            obs_encoding_flat = self.obs_encoder.extract_features(obs_img) # obs_encoding shape: torch.Size([512, 384]) 
            
            # Process goal encoding based on encoder type
            if self.goal_encoder_type == "image_pair" or self.goal_encoder_type == "image_pair_position":
                # Original image-based processing
                if goal_img is None:
                    raise ValueError("goal_img must be provided for goal_image encoder type")
                obsgoal_img = torch.cat(
                    [obs_img[:, 3 * self.context_size:, :, :], goal_img], dim=0
                )  # concatenate the obs image/context and goal image
                obsgoal_encoding = self.goal_encoder.extract_features(obsgoal_img)

                # concat the features
                obsgoal_encoding = obsgoal_encoding.reshape(2, -1, obsgoal_encoding.shape[-1])
                obsgoal_encoding = obsgoal_encoding.transpose(0, 1)
                
                if self.goal_encoder_type == "image_pair_position":
                    obsgoal_encoding = self.goal_gmc.encode_forward("joint", x=[obsgoal_encoding, goal_pos])
                else:
                    obsgoal_encoding = self.goal_gmc.encode_forward("features", x=obsgoal_encoding)

            elif self.goal_encoder_type == "position":
                # Process goal positions using dummy encoder
                if goal_pos is None:
                    raise ValueError("goal_pos must be provided for position encoder type")
                obsgoal_encoding = self.goal_gmc.encode_forward("coord", x=goal_pos)

        else:
            # Pre-built feature processing
            # obs_img shape: [batch_size, context_size+1, feature_dim]

            # Process observation features
            batch_size = obs_img.shape[0]

            # Compress observation features
            obs_encoding_flat = obs_img.reshape(-1, obs_img.shape[-1])  # Flatten to [batch_size*(context_size+1), feature_dim]

            # Process goal features based on encoder type
            if self.goal_encoder_type == "feature_pair":
                # Original goal image processing
                if goal_img is None:
                    raise ValueError("goal_img must be provided for goal_image encoder type")
                # concat current features with goal features
                obsgoal_encoding = torch.cat([obs_img[:, -1, :], goal_img], dim=-1)
                obsgoal_encoding = self.goal_gmc.encode_forward("features", x=obsgoal_encoding)

            elif self.goal_encoder_type == "position":
                # Process goal positions using dummy encoder
                if goal_pos is None:
                    raise ValueError("goal_pos must be provided for position encoder type")
                obsgoal_encoding = self.goal_gmc.encode_forward("coord", x=goal_pos)

            elif self.goal_encoder_type == "feature_pair_position":
                # Process image pairs + goal positions using dummy encoder
                if goal_img is None or goal_pos is None:
                    raise ValueError("Both goal_img and goal_pos must be provided for feature_pair_position encoder type")
                pair_features = torch.cat([obs_img[:, -1, :], goal_img], dim=-1)
                obsgoal_encoding = self.goal_gmc.encode_forward("joint", x=[pair_features, goal_pos])
        
        # compress and reshape obs encodings
        obs_encoding = self.compress_obs_enc(obs_encoding_flat) # obs_encoding shape: torch.Size([512, 256])
        obs_encoding = obs_encoding.unsqueeze(1) # obs_encoding shape: torch.Size([512, 1, 256])
        obs_encoding = obs_encoding.reshape((self.context_size + 1, -1, self.obs_encoding_size)) # obs_encoding shape: torch.Size([4, 128, 256])
        obs_encoding = torch.transpose(obs_encoding, 0, 1) # obs_encoding shape: torch.Size([128, 4, 256])
        
        # compress and reshape goal encodings
        obsgoal_encoding = self.compress_goal_enc(obsgoal_encoding)  # Shape: [batch_size, encoding_size]
        obsgoal_encoding = obsgoal_encoding.unsqueeze(1)  # Shape: [batch_size, 1, encoding_size]
        
        # Concatenate observation and goal encodings
        obs_encoding = torch.cat((obs_encoding, obsgoal_encoding), dim=1) # obs_encoding shape: torch.Size([128, 5, 256])

        # If a goal mask is provided, mask some of the goal tokens
        if input_goal_mask is not None:
            goal_mask = input_goal_mask.to(device)
            no_goal_mask = goal_mask.long()
            src_key_padding_mask = torch.index_select(
                self.all_masks.to(device), 0, no_goal_mask
            )
        else:
            src_key_padding_mask = None

        # Apply positional encoding
        if self.positional_encoding:
            obs_encoding = self.positional_encoding(obs_encoding)

        obs_encoding_tokens = self.sa_encoder(
            obs_encoding, src_key_padding_mask=src_key_padding_mask
        )
        if src_key_padding_mask is not None:
            avg_mask = torch.index_select(
                self.avg_pool_mask.to(device), 0, no_goal_mask
            ).unsqueeze(-1)
            obs_encoding_tokens = obs_encoding_tokens * avg_mask
        obs_encoding_tokens = torch.mean(obs_encoding_tokens, dim=1)

        return obs_encoding_tokens
