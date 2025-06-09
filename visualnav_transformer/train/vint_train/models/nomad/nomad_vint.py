from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

from visualnav_transformer.train.vint_train.models.vint.self_attention import (
    PositionalEncoding,
)
from visualnav_transformer.train.vint_train.models.nomad.ft_extractor import (
    EfficientNetExtractor, DiNOV2Extractor
)


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
        """
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size
        self.use_prebuilt_features = use_prebuilt_features
        
        if not use_prebuilt_features:
            print(f"Using feature extractor: {obs_encoder.split('-')[0]}")
            
            if obs_encoder.split("-")[0] == "efficientnet":
                self.obs_encoder = EfficientNetExtractor(obs_encoder, in_channels=3)
                self.goal_encoder = EfficientNetExtractor("efficientnet-b0", in_channels=6)
            elif obs_encoder.split("-")[0] == "dinov2":
                self.obs_encoder = DiNOV2Extractor(obs_encoder)
                self.goal_encoder = DiNOV2Extractor(obs_encoder)
            self.num_obs_features = self.obs_encoder.num_out_features
            self.num_goal_features = 2 * self.goal_encoder.num_out_features
        else:
            print("Using pre-built DINO features")
            # When using pre-built features, we need to know the feature dimensions
            # These will be determined by the DINO model used to extract features
            if "large" in obs_encoder:
                self.num_obs_features = 1024
                self.num_goal_features = 1024
            elif "base" in obs_encoder:
                self.num_obs_features = 768
                self.num_goal_features = 768
            elif "small" in obs_encoder:
                self.num_obs_features = 384
                self.num_goal_features = 384
            else:
                # Default to small
                self.num_obs_features = 384
                self.num_goal_features = 384

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
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size,
            nhead=mha_num_attention_heads,
            dim_feedforward=mha_ff_dim_factor * self.obs_encoding_size,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sa_encoder = nn.TransformerEncoder(
            self.sa_layer, num_layers=mha_num_attention_layers
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
        goal_img: torch.tensor, # Either images [128, 3, 240, 320] or pre-computed features [batch_size, feature_dim]
        input_goal_mask: torch.tensor = None, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = obs_img.device

        # Get the input goal mask
        if input_goal_mask is not None:
            goal_mask = input_goal_mask.to(device)
        
        if not self.use_prebuilt_features:
            # Original image-based processing
            # Get the goal encoding
            obsgoal_img = torch.cat(
                [obs_img[:, 3 * self.context_size:, :, :], goal_img], dim=0
            )  # concatenate the obs image/context and goal image --> non image goal? - obsgoal_img shape: torch.Size([256, 3, 240, 320])
            obsgoal_encoding = self.goal_encoder.extract_features(obsgoal_img) # obsgoal_encoding: [256, 384] (128 observation-goal pairs concatenated along batch dim)
            
            obsgoal_encoding = obsgoal_encoding.reshape(2, -1, obsgoal_encoding.shape[-1])  # seperate latent encoded (obs-goal) pair to [2, 128, 384]
            obsgoal_encoding = obsgoal_encoding.transpose(0, 1) # transpose dimensions to put batch size first, [128, 2, 384]   
            obsgoal_encoding = obsgoal_encoding.reshape(obsgoal_encoding.shape[0], -1) # concatenate pair encodings along latent dimension, [128, 768]
            obsgoal_encoding = self.compress_goal_enc(obsgoal_encoding) # join pair dino compress - [128, 256]
            
            # Changes Here with feature extractor dimensions !!! - Cimbi
            
            if len(obsgoal_encoding.shape) == 2:
                obsgoal_encoding = obsgoal_encoding.unsqueeze(1)
            assert obsgoal_encoding.shape[2] == self.goal_encoding_size

            # Get the observation encoding
            obs_img = torch.split(obs_img, 3, dim=1) # obs_img shape: 4
            obs_img = torch.concat(obs_img, dim=0) # obs_img shape: 512
            obs_encoding = self.obs_encoder.extract_features(obs_img) # obs_encoding shape: torch.Size([512, 384]) 
            obs_encoding = self.compress_obs_enc(obs_encoding) # obs_encoding shape: torch.Size([512, 256])
            obs_encoding = obs_encoding.unsqueeze(1) # obs_encoding shape: torch.Size([512, 1, 256])
            obs_encoding = obs_encoding.reshape(
                (self.context_size + 1, -1, self.obs_encoding_size)
            ) # obs_encoding shape: torch.Size([4, 128, 256])
            obs_encoding = torch.transpose(obs_encoding, 0, 1) # obs_encoding shape: torch.Size([128, 4, 256])
        else:
            # Pre-built feature processing
            # obs_img shape: [batch_size, context_size+1, feature_dim]
            # goal_img shape: [batch_size, feature_dim]
            
            # Process observation features
            batch_size = obs_img.shape[0]
            
            # Compress observation features
            obs_features_flat = obs_img.reshape(-1, obs_img.shape[-1])  # Flatten to [batch_size*(context_size+1), feature_dim]
            obs_encoding_flat = self.compress_obs_enc(obs_features_flat)  # Shape: [batch_size*(context_size+1), encoding_size]
            obs_encoding = obs_encoding_flat.reshape(batch_size, self.context_size + 1, self.obs_encoding_size)  # Shape: [batch_size, context_size+1, encoding_size]
            
            # Process goal features
            goal_feature_compressed = self.compress_goal_enc(goal_img)  # Shape: [batch_size, encoding_size]
            obsgoal_encoding = goal_feature_compressed.unsqueeze(1)  # Shape: [batch_size, 1, encoding_size]
        
        # Concatenate observation and goal encodings
        obs_encoding = torch.cat((obs_encoding, obsgoal_encoding), dim=1) # obs_encoding shape: torch.Size([128, 5, 256])

        # If a goal mask is provided, mask some of the goal tokens
        if goal_mask is not None:
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
