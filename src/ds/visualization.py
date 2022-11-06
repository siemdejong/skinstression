import logging
import os

import optuna
from optuna.visualization import (
    plot_contour,
    plot_intermediate_values,
    plot_parallel_coordinate,
    plot_param_importances,
)

from conf.config import THGStrainStressConfig


def visualize(cfg: THGStrainStressConfig):
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
