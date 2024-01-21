"""Skinstression: skin stretch regression using deep learning
	Copyright (C) 2024  Siem de Jong
    See LICENSE for full license.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

cli_license_notice = """
	Skinstression for skin stretch curve regression.
	Copyright (C) 2024  Siem de Jong

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
    """


def logistic(x, a, k, xc):
    return a / (1 + np.exp(-(x - xc) * k))


def plot_prediction(
    pred: npt.NDArray,
    target: npt.NDArray,
    slice_idx=None,
    num_slices=None,
):
    cmap = mpl.colormaps["viridis"].resampled(num_slices)

    x = np.linspace(1, 1.5, 100)
    if len(pred) == 3:
        plt.plot(x, logistic(x, pred[0], pred[1], pred[2]), c=cmap(abs(slice_idx)))
    else:
        raise NotImplementedError(
            "Plotting with only one target is not implemented yet."
        )

    plt.plot(x, logistic(x, target[0], target[1], target[2]), c="red")
