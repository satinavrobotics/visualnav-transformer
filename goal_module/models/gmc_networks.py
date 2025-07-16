import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import CoordEncoder, ImagePairEncoder

# COMMON

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    

class SharedProjection(nn.Module):

    def __init__(self, common_dim, latent_dim):
        super(SharedProjection, self).__init__()
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(common_dim, 512),
            nn.LayerNorm(512),
            Swish(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            Swish(),
            nn.Linear(512, latent_dim),
        )

        # Initialize weights
        self._init_weights()

    def forward(self, x):
        return F.normalize(self.feature_extractor(x), dim=-1)

    def _init_weights(self):
        """Initialize weights using Kaiming initialization for Swish activations."""
        for module in self.feature_extractor:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')  # Swish behaves similarly to ReLU
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    

class ImageFeaturePairProcessor(nn.Module):

    def __init__(self, common_dim, num_heads=8, dropout=0.1):
        super(ImageFeaturePairProcessor, self).__init__()
        self.common_dim = common_dim

        # Use the new ImagePairEncoder with attention-based fusion
        self.image_pair_encoder = ImagePairEncoder(
            output_dim=common_dim,
            fusion_type="attention",
            num_heads=num_heads,
            dropout=dropout,
            feature_dim=1024
        )

        self.loss_name = "feature_repr_loss"

    def forward(self, x):
        # x shape: [batch_size, 2, 1024]
        return self.image_pair_encoder(x)
    


class CoordProcessor(nn.Module):
    def __init__(self, common_dim):
        super(CoordProcessor, self).__init__()
        self.common_dim = common_dim

        # Process goal position [3] -> [common_dim]
        self.coord_encoder = CoordEncoder(
            fourier_sigma=10.0,
            fourier_m=16,
            coord_dim=4,
            hidden_dim=128,
            output_dim=64,
            depth=4,
            use_layernorm=True,
        )
        self.projector = nn.Linear(64, common_dim)
        self.loss_name = "coord_repr_loss"

        # Initialize weights
        self._init_weights()

    def forward(self, x):
        # x shape: [batch_size, 4]
        h = self.coord_encoder(x)  # [batch_size, 64]
        return self.projector(h)  # [batch_size, common_dim]

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        # Initialize the projector (no activation after this layer, so Xavier is appropriate)
        nn.init.xavier_uniform_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)

        # Note: CoordEncoder initialization should be handled in its own class


class JointProcessor(nn.Module):
    def __init__(self, common_dim):
        super(JointProcessor, self).__init__()
        self.common_dim = common_dim

        # Use the new ImagePairEncoder with MLP-based fusion
        self.image_features_processor = ImagePairEncoder(
            output_dim=common_dim,
            fusion_type="mlp",
            feature_dim=1024
        )

        self.position_processor = CoordEncoder(
            fourier_sigma=10.0,
            fourier_m=16,
            coord_dim=4,
            hidden_dim=128,
            output_dim=64,
            depth=6,
            use_layernorm=True,
        )

        # Joint projection
        self.joint_projector = nn.Linear(common_dim + 64, common_dim)
        self.loss_name = "joint_repr_loss"

        # Initialize weights
        self._init_weights()

    def forward(self, x):
        # x is a list: [image_features, goal_position]
        image_features, goal_position = x[0], x[1]
        # image_features should be [batch_size, 2, 1024] for ImagePairEncoder

        # Process each modality
        h_image_features = self.image_features_processor(image_features)
        h_goal_position = self.position_processor(goal_position)

        # Concatenate all representations
        joint_repr = torch.cat([h_image_features, h_goal_position], dim=-1)

        return self.joint_projector(joint_repr)

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        # ImagePairEncoder handles its own initialization

        # Initialize joint projector
        nn.init.xavier_uniform_(self.joint_projector.weight)
        nn.init.zeros_(self.joint_projector.bias)

        # Note: CoordEncoder initialization should be handled in its own class

