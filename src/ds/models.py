"""Provides trainable Pytorch models.
Copyright (C) 2022  Siem de Jong

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import torch
from torch import nn

from conf.config import SkinstressionConfig


class SkinstressionCNN(nn.Module):
    """Convolutional Neural Network (CNN) to calculate strain-stress features in SHG skin images.

    Assumes a 2D RGB 258*258*3 input image.
    """

    def __init__(self, cfg: SkinstressionConfig) -> None:
        super(SkinstressionCNN, self).__init__()

        layers = []

        # Pre-blocks
        # for _ in range(14):
        #     layers.append(nn.Conv2d(1, 1, 53, bias=False))
        #     layers.append(nn.BatchNorm2d(1))
        #     layers.append(nn.ReLU())

        # Block 1
        layers.append(nn.Conv2d(1, 64, 3, bias=False))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())
        # layers.append(nn.Dropout2d(cfg.params.model.dropout_1))
        for _ in range(3):
            layers.append(nn.MaxPool2d(2))

        # Block 2
        layers.append(nn.Conv2d(64, 64, 5, bias=False))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())
        # layers.append(nn.Dropout2d(cfg.params.model.dropout_2))
        layers.append(nn.MaxPool2d(2))

        # Block 3
        layers.append(nn.Conv2d(64, 64, 3, bias=False))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())
        # layers.append(nn.Dropout2d(cfg.params.model.dropout_3))
        layers.append(nn.MaxPool2d(2))

        # Block 4
        layers.append(nn.Conv2d(64, 64, 6, bias=False))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())
        # layers.append(nn.Dropout2d(cfg.params.model.dropout_4))

        # MLP
        layers.append(nn.Flatten())
        layers.append(nn.Linear(64, cfg.params.model.n_nodes, bias=False))
        layers.append(nn.BatchNorm1d(cfg.params.model.n_nodes))
        layers.append(nn.ReLU())
        layers.append(
            nn.Linear(cfg.params.model.n_nodes, cfg.params.model.num_output_features)
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
