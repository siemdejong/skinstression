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

import numpy as np
import warnings

# This warning happens when -k * (x - xc) is large.
# This is unavoidable, since the neural network predicts
# these values and might end up with extreme values.
warnings.filterwarnings(
    "ignore",
    message="overflow encountered in exp",
    category=RuntimeWarning,
)


def sigmoid(x, A, h, slope, C) -> float:
    """https://stackoverflow.com/a/55104465"""
    return 1 / (1 + np.exp((x - h) / slope)) * A + C


def logistic(x, a, k, xc):
    return a / (1 + np.exp(-k * (x - xc)))
