# Image Model class

import argparse
from typing import Any, Dict

import torch.nn as nn

from torchvision import models

# Not required
FC1_DIM = 1
FC2_DIM = 1
FC_DROPOUT = 1


class resnet18(nn.Module):
    """Pretrained resnet model"""

    def __init__(
        self,
        data_config: Dict[str, Any] = None,
        args: argparse.Namespace = None,
    ) -> None:

        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.data_config = data_config

        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

        

    def forward(self, x):
        x = self.model(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
      
       return parser