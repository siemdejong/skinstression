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

log = logging.getLogger(__name__)


def visualize(database: str):
    """Visualize Optuna optimization output.
    Plots are opened in an external browser at ports opened by Plotly.

    Args:
        cfg: hydra configuration object. cfg.paths.optuna_db must be provided.
    """
    study_name = os.path.basename(database).split(".")[0]
    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{database}",
    )

    intermediate_values = plot_intermediate_values(study)
    intermediate_values.show()

    contour = plot_contour(study)
    contour.show()

    param_importances = plot_param_importances(study)
    param_importances.show()

    parallel_coordinate = plot_parallel_coordinate(study)
    parallel_coordinate.show()

    log.info("A browser should have been opened, showing the plots")
