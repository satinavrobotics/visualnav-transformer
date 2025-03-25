import math
import itertools
from functools import partial

import torch
import torch.nn.functional as F

from dinov2.eval.depth.models import build_depther



class DiNOV2(nn.Module):
    
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    
    def __init__(self, backbone_size: str = "small"):
        super(DiNOV2, self).__init__()
        
        self.backbone_size = backbone_size
        self.backbone_arch = backbone_archs[BACKBONE_SIZE]
        self.backbone_name = f"dinov2_{backbone_arch}"

        backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)

    def forward(self, batch):
        return model.whole_inference(batch, img_meta=None, rescale=True)
    
    def extract_features(self, batch):
        return self.forward(batch)







