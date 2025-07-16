import torch
import torch.nn as nn
import torch.nn.functional as F


class NoMaD(nn.Module):

    def __init__(self, vision_encoder, noise_pred_net, dist_pred_net, pose_head):
        super(NoMaD, self).__init__()

        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
        self.dist_pred_net = dist_pred_net
        self.pose_head = pose_head

    def forward(self, func_name, **kwargs):
        if func_name == "vision_encoder":
            output = self.vision_encoder(
                kwargs["obs_img"],
                kwargs["goal_img"],
                input_goal_mask=kwargs["input_goal_mask"],
            )
        elif func_name == "noise_pred_net":
            output = self.noise_pred_net(
                sample=kwargs["sample"],
                timestep=kwargs["timestep"],
                global_cond=kwargs["global_cond"],
            )
        elif func_name == "dist_pred_net":
            output = self.dist_pred_net(kwargs["obsgoal_cond"])
            
        elif func_name == "pose_head":
            output = self.pose_head(kwargs["obsgoal_cond"])
        else:
            raise NotImplementedError
        return output


class DenseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(DenseNetwork, self).__init__()

        self.embedding_dim = embedding_dim
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 4, self.embedding_dim // 16),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 16, 1),
        )

    def forward(self, x):
        x = x.reshape((-1, self.embedding_dim))
        output = self.network(x)
        return output


class PoseHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        # Shared projection (optional, can be removed if unnecessary)
        self.shared = nn.Linear(input_dim, hidden_dim)

        # Shallow position head
        self.pos_head = nn.Linear(hidden_dim, 2)

        # Shallow yaw head with normalization
        self.yaw_head = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = F.relu(self.shared(x))

        pos = self.pos_head(x)
        yaw = self.yaw_head(x)
        yaw = F.normalize(yaw, dim=1)  # enforce sin²+cos² = 1

        return torch.cat([pos, yaw], dim=1)  # [x, y, sin(yaw), cos(yaw)]