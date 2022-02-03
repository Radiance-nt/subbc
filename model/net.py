import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, out_dim),
        )

    def forward(self, obs):
        return self.mlp(obs)