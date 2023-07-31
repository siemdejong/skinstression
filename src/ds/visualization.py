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
from pathlib import Path
import pandas as pd

import optuna
from optuna.visualization import (
    plot_contour,
    plot_intermediate_values,
    plot_parallel_coordinate,
    plot_param_importances,
)

from conf.config import SkinstressionConfig

import plotly
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def custom_parallel_coordinates(df, params, names):
    """Plots parallel coordinates from an Optuna trials df.
    
    TODO: Does not yet use the params/names arguments. Only displays selective df columns.
    """
    fig = go.Figure(
        data=go.Parcoords(
            line = dict(
                color = np.log10(df['value']),
                colorscale = px.colors.diverging.balance,
                showscale = True,
                # cmid = 0,
                # colorbar=dict(
                #               x=0.8,
                #               tickvals=np.log10(tickvals),
                #               ticktext=tickvals,
                #           ),
                # cmin = -4000,
                # cmax = -100,
            ),
            dimensions = list([
                dict(
                    # range = [32000,227900],
                    # constraintrange = [100000,150000],
                    label = "log lowest loss", values = np.log10(df['value'])
                ),
                dict(
                    # range = [32000,227900],
                    # constraintrange = [100000,150000],
                    label = "T0", values = df['params_T_0']
                ),
                dict(
                    # range = [0,700000],
                    label = 'log lr', values = np.log10(df['params_lr'])
                ),
                dict(
                    # tickvals = [0,0.5,1,2,3],
                    # ticktext = ['A','AB','B','Y','Z'],
                    label = 'Tmult', values = df['params_T_mult']
                ),
                dict(
                    # range = [-1,4],
                    # tickvals = [0,1,2,3],
                    label = 'log weight decay', values = np.log10(df['params_weight_decay'])
                ),
                dict(
                    # range = [-1,4],
                    # tickvals = [0,1,2,3],
                    label = 'batch size', values = df['params_batch_size']
                ),
                dict(
                    # range = [-1,4],
                    # tickvals = [0,1,2,3],
                    label = 'Nodes', values = df['params_n_nodes']
                ),
            ]),
            ),
        )
    return fig



def visualize(cfg: SkinstressionConfig):
    """Visualize Optuna optimization output.
    Plots are opened in an external browser at ports opened by Plotly.

    Args:
        cfg: hydra configuration object. cfg.paths.optuna_db must be provided.
    """

    Path(f'optuna/{cfg.optuna.study_name}/').mkdir(parents=True, exist_ok=True)

    study = optuna.load_study(
        study_name=cfg.optuna.study_name,
        storage=f"sqlite:///{cfg.paths.optuna_db}",
    )
    trials_df = study.trials_dataframe()
    
    # iterating the columns to see what data is actually in there :)
    # for col in trials_df.columns:
    #     print(col)
    # exit()

    params = list(optuna.importance.get_param_importances(study).keys())

    intermediate_values = plot_intermediate_values(study)
    intermediate_values.update_layout(yaxis_type='log')
    intermediate_values.show()
    plotly.io.write_image(intermediate_values, f'optuna/{cfg.optuna.study_name}/intermediate_values.pdf', format='pdf')
    logging.info("Plotting intermediate values.")

    trials = study.trials_dataframe(attrs=("state",))
    num_completed = len(trials[trials["state"] == "COMPLETE"])
    if num_completed > 1:
        contour = plot_contour(study)
        contour.show()
        plotly.io.write_image(contour, f'optuna/{cfg.optuna.study_name}/contour.pdf', format='pdf')
        logging.info("Plotting parameter contours.")

        param_importances = plot_param_importances(study)
        param_importances.show()
        plotly.io.write_image(param_importances, f'optuna/{cfg.optuna.study_name}/param_importances.pdf', format='pdf')
        logging.info("Plotting parameter importances.")

        parallel_coordinate = custom_parallel_coordinates(trials_df, params, [' '.join(param.split('_')[1:]) for param in params])
        plotly.io.write_image(parallel_coordinate, f'optuna/{cfg.optuna.study_name}/parallel_coordinate.pdf', format='pdf')
        logging.info("Plotting parallel coordinates.")
    else:
        print(
            "For more plots to show, "
            "please ensure more trials have the COMPLETE state."
        )

    logging.info("A browser should have been opened.")

