from conf.config import THGStrainStressConfig
import os
import logging
import optuna
from optuna.visualization import (
    plot_intermediate_values,
    plot_contour,
    plot_param_importances,
    plot_parallel_coordinate,
)

log = logging.getLogger(__name__)


def visualize(cfg: THGStrainStressConfig):
    """Visualize Optuna optimization output.
    Plots are opened in an external browser at ports opened by Plotly.

    Args:
        cfg: hydra configuration object. cfg.paths.optuna_db must be provided.
    """

    database = cfg.paths.optuna_db
    study_name = os.path.basename(database).split(".")[0]
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{database}",
        load_if_exists=True,
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
