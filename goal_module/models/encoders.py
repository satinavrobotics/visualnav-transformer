import torch
import torch.nn as nn
import torch.nn.functional as F
import rff.layers as rff


class CoordEncoder(nn.Module):
    def __init__(
        self,
        fourier_sigma: float = 10.0,
        fourier_m: int = 16,
        coord_dim: int = 4,        # Δx, Δy, Δsin(θ), Δcos(θ)
        hidden_dim: int = 1024,
        output_dim: int = 512,
        depth: int = 6,
        use_layernorm: bool = True,
    ):
        super().__init__()
        # Fourier-features
        self.fourier = rff.PositionalEncoding(sigma=fourier_sigma, m=fourier_m)
        fourier_dim = coord_dim * 2 * fourier_m
        
        # Input hosszabbítás: Fourier + nyers coord
        self.input_dim = fourier_dim + coord_dim
        self.blocks = nn.ModuleList()
        for i in range(depth):
            in_dim = self.input_dim if i == 0 else hidden_dim
            lin = nn.Linear(in_dim, hidden_dim)
            norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
            act = nn.GELU()
            # Residual kapcsolat: blokk input + output
            block = nn.ModuleDict({'lin': lin, 'norm': norm, 'act': act})
            self.blocks.append(block)
        
        self.final = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        self._init_weights()

    def forward(self, coord: torch.Tensor):
        # coord: [batch, 3] – relatív x, y, θ (normalizált)
        x_fourier = self.fourier(coord)
        h = torch.cat([x_fourier, coord], dim=-1)
        
        for block in self.blocks:
            y = block['lin'](h)
            y = block['norm'](y)
            y = block['act'](y)
            # Residual kapcsolat (skip): h lehet input vagy hidden
            if y.shape == h.shape:
                h = h + y
            else:
                h = y  # csak első réteg esetén, ha dim változik
        return self.final(h)

    def _init_weights(self):
        """Initialize weights using Kaiming initialization for GELU activations."""
        # Initialize all linear layers in blocks
        for block in self.blocks:
            lin_layer = block['lin']
            nn.init.kaiming_uniform_(lin_layer.weight, nonlinearity='relu')  # GELU behaves similarly to ReLU
            nn.init.zeros_(lin_layer.bias)

            # Layer normalization parameters (if present)
            if hasattr(block['norm'], 'weight'):
                nn.init.ones_(block['norm'].weight)
                nn.init.zeros_(block['norm'].bias)

        # Initialize final layer (no activation after this, so Xavier is appropriate)
        nn.init.xavier_uniform_(self.final.weight)
        nn.init.zeros_(self.final.bias)


class ImagePairEncoder(nn.Module):
    """
    Encoder for processing pairs of image features.
    Supports both attention-based and MLP-based fusion strategies.
    """

    def __init__(self, output_dim, fusion_type="attention", num_heads=8, dropout=0.1, feature_dim=1024):
        super(ImagePairEncoder, self).__init__()
        self.output_dim = output_dim
        self.feature_dim = feature_dim
        self.fusion_type = fusion_type

        if fusion_type == "attention":
            # Multihead attention for feature fusion
            self.multihead_attn = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            # Layer normalization
            self.layer_norm = nn.LayerNorm(feature_dim)
            # Single layer projection
            self.projector = nn.Linear(feature_dim, output_dim)

        elif fusion_type == "mlp":
            # Simple MLP-based fusion (concatenation + MLP)
            self.projector = nn.Sequential(
                nn.Linear(2 * feature_dim, 512),
                nn.GELU(),
                nn.Linear(512, output_dim),
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. Use 'attention' or 'mlp'.")

        # Initialize weights
        self._init_weights()

    def forward(self, x):
        """
        Forward pass for image pair encoding.

        Args:
            x: Tensor of shape [batch_size, 2, feature_dim] containing pairs of image features

        Returns:
            Tensor of shape [batch_size, output_dim] containing fused features
        """
        if self.fusion_type == "attention":
            x = x.view(x.size(0), 2, -1)
            # Apply multihead attention (self-attention between the two features)
            attn_output, attn_weights = self.multihead_attn(x, x, x)

            # Apply layer normalization
            attn_output = self.layer_norm(attn_output)

            # Pool the attended features (mean pooling across the 2 features)
            pooled_features = attn_output.mean(dim=1)  # [batch_size, feature_dim]

            # Single layer projection to output_dim
            output = self.projector(pooled_features)

        elif self.fusion_type == "mlp":
            # Flatten and concatenate the features
            x_flat = x.view(x.size(0), -1)  # [batch_size, 2*feature_dim]

            # Apply MLP
            output = self.projector(x_flat)

        return output

    def _init_weights(self):
        """Initialize weights using Kaiming initialization for GELU activations."""
        if self.fusion_type == "attention":
            # Initialize the projector linear layer (no activation after this, so Xavier is fine)
            nn.init.xavier_uniform_(self.projector.weight)
            nn.init.zeros_(self.projector.bias)

            # Initialize multihead attention weights (no GELU in attention, so Xavier is appropriate)
            for name, param in self.multihead_attn.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

            # Layer normalization parameters
            nn.init.ones_(self.layer_norm.weight)
            nn.init.zeros_(self.layer_norm.bias)

        elif self.fusion_type == "mlp":
            # Initialize MLP layers with Kaiming initialization for GELU activations
            for module in self.projector:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')  # GELU behaves similarly to ReLU
                    nn.init.zeros_(module.bias)