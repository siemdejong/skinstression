"""Provides equations as functions.
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

import math
import torch


@torch.jit.script
def sigmoid(x, A, h, slope, C) -> float:
    """https://stackoverflow.com/a/55104465"""
    return 1 / (1 + math.exp((x - h) / slope)) * A + C


@torch.jit.script  # To fuse pointwise operations making calculation quicker.
def logistic(x, a, k, xc):
    return a / (1 + math.exp(-k * (x - xc)))
