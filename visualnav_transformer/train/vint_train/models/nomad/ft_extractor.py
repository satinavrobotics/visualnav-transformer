import math
import itertools
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet



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
        
        self.backbone_size = backbone_size
        self.backbone_arch = self.backbone_archs[backbone_size]
        self.num_out_features = self.backbone_ft_sizes[backbone_size]
        self.backbone_name = f"dinov2_{self.backbone_arch}"

        self.backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=self.backbone_name)
        
        # Freeze the weights
        for param in self.backbone_model.parameters():
            param.requires_grad = False

    def forward(self, batch):
        return self.backbone_model.whole_inference(batch, img_meta=None, rescale=True)
    
    def extract_features(self, batch):
        return self.forward(batch)
    
    def num_out_features(self):
        return self.num_out_features
    
    
class EfficientNetExtractor(nn.Module):
    def __init__(self, model_name: str = "efficientnet-b0", in_channels: int = 3):
        super(EfficientNet, self).__init__()
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








