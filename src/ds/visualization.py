"""Provides visualization function to visualize hyperparameter optimization.
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

import logging

import optuna
from optuna.visualization import (
    plot_contour,
    plot_intermediate_values,
    plot_parallel_coordinate,
    plot_param_importances,
)

from conf.config import SkinstressionConfig


def visualize(cfg: SkinstressionConfig):
    """Visualize Optuna optimization output.
    Plots are opened in an external browser at ports opened by Plotly.

    Args:
        cfg: hydra configuration object. cfg.paths.optuna_db must be provided.
    """
    study = optuna.load_study(
        study_name=cfg.optuna.study_name,
        storage=f"sqlite:///{cfg.paths.optuna_db}",
    )

    intermediate_values = plot_intermediate_values(study)
    intermediate_values.show()
    logging.info("Plotting intermediate values.")

    trials = study.trials_dataframe(attrs=("state",))
    num_completed = len(trials[trials["state"] == "COMPLETE"])
    if num_completed > 1:
        contour = plot_contour(study)
        contour.show()
        logging.info("Plotting parameter contours.")

        param_importances = plot_param_importances(study)
        param_importances.show()
        logging.info("Plotting parameter importances.")

        parallel_coordinate = plot_parallel_coordinate(study)
        parallel_coordinate.show()
        logging.info("Plotting hyperparameter network.")
    else:
        print(
            "For more plots to show, "
            "please ensure more trials have the COMPLETE state."
        )

    logging.info("A browser should have been opened.")
