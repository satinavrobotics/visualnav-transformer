import math
import itertools
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

#from efficientnet_pytorch import EfficientNet



class DiNOV2Extractor(nn.Module):

    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }

    backbone_ft_sizes = {
        "small": 384,
        "base": 768,
        "large": 1024,
        "giant": 1536,
    }

    def __init__(self, backbone_size: str = "small"):
        super(DiNOV2Extractor, self).__init__()  # Corrected the class name

        if backbone_size.startswith("dinov2-"):
            backbone_size = backbone_size.split("-")[1]  # extract "small", "base", etc.

        self.backbone_size = backbone_size
        self.backbone_arch = self.backbone_archs[backbone_size]
        self.num_out_features = self.backbone_ft_sizes[backbone_size]
        self.backbone_name = f"dinov2_{self.backbone_arch}"

        self.backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=self.backbone_name)

        # Freeze the weights
        for param in self.backbone_model.parameters():
            param.requires_grad = False

        # Define required divisible patch size
        self.patch_size = 14

    # F is already imported at the top

    def pad_to_divisible(self, img_tensor):
        """
        Makes tensor exactly (238, 322) from  input (240, 320) for Dino V2 input kernel
        img_tensor: [B, C, H, W] - Original shape: torch.Size([256, 3, 240, 320])
        """

        # Pad width: add 1 column on each side (total 2 columns)
        img_tensor = F.pad(img_tensor, (1, 1, 0, 0), mode='constant', value=0) # After padding columns: torch.Size([256, 3, 240, 322])
        # Crop height: remove 1 row from top and 1 from bottom (total 2 rows)
        img_tensor = img_tensor[:, :, 1:-1, :] # After cropping rows: torch.Size([256, 3, 238, 322])

        return img_tensor

    def forward(self, x):
        """
        Forward pass that handles channel adjustment and padding automatically.
        """
        x = self.pad_to_divisible(x)
        features = self.backbone_model(x)
        return features

    def extract_features(self, batch):
        return self.forward(batch)

    def extract_features_batch(self, images, batch_size=32, device=None):
        """Extract features from a list of images in batches."""
        if device is None:
            device = next(self.parameters()).device

        all_features = []
        for i in range(0, len(images), batch_size):
            batch_images = torch.stack(images[i:i+batch_size]).to(device)
            with torch.no_grad():
                batch_features = self.extract_features(batch_images)
            all_features.append(batch_features.cpu())

        if all_features:
            return torch.cat(all_features, dim=0)
        else:
            return None

    def num_out_features(self):
        return self.num_out_features


class EfficientNetExtractor(nn.Module):
    def __init__(self, model_name: str = "efficientnet-b0", in_channels: int = 3):
        super(EfficientNetExtractor, self).__init__()
        # Initialize the observation encoder
        if model_name.split("-")[0] == "efficientnet":
            self.encoder = EfficientNet.from_name(
                model_name, in_channels=in_channels
            )  # context
            self.encoder = replace_bn_with_gn(self.encoder)
            self.num_obs_features = self.encoder._fc.in_features
        else:
            raise NotImplementedError

    def extract_features(self, x):
        x = self.encoder.extract_features(x)  # get encoding of this img
        x = self.encoder._avg_pooling(x)  # avg pooling

        if self.encoder._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self.encoder._dropout(x)

        return x

    def forward(self, x):
        return self.encoder(x)

    def num_out_features(self):
        return self.num_obs_features








