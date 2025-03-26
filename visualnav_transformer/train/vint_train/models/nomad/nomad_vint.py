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
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        """
        NoMaD ViNT Encoder class
        """
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size
        
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNetExtractor(obs_encoder, in_channels=3)
            self.goal_encoder = EfficientNetExtractor("efficientnet-b0", in_channels=6)
        elif obs_encoder.split("-")[0] == "dinov2":
            self.obs_encoder = DiNOV2Extractor(obs_encoder)
            self.goal_encoder = DiNOV2Extractor(obs_encoder)
        self.num_obs_features = self.obs_encoder.num_out_features()
        self.num_goal_features = self.goal_encoder.num_out_features()

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
        obs_img: torch.tensor,
        goal_img: torch.tensor,
        input_goal_mask: torch.tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = obs_img.device

        # Get the input goal mask
        if input_goal_mask is not None:
            goal_mask = input_goal_mask.to(device)

        # Get the goal encoding
        obsgoal_img = torch.cat(
            [obs_img[:, 3 * self.context_size :, :, :], goal_img], dim=1
        )  # concatenate the obs image/context and goal image --> non image goal?
        obsgoal_encoding = self.goal_encoder.extract_features(obsgoal_img)
        obsgoal_encoding = self.compress_goal_enc(obsgoal_encoding)

        if len(obsgoal_encoding.shape) == 2:
            obsgoal_encoding = obsgoal_encoding.unsqueeze(1)
        assert obsgoal_encoding.shape[2] == self.goal_encoding_size

        # Get the observation encoding
        obs_img = torch.split(obs_img, 3, dim=1)
        obs_img = torch.concat(obs_img, dim=0)
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.unsqueeze(1)
        obs_encoding = obs_encoding.reshape(
            (self.context_size + 1, -1, self.obs_encoding_size)
        )
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        obs_encoding = torch.cat((obs_encoding, obsgoal_encoding), dim=1)

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

