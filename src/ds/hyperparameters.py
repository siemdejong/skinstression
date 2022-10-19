from typing import Any
from conf.config import THGStrainStressConfig
from ds.dataset import THGStrainStressDataset
from ds.runner import Runner, Stage, run_epoch
from ds.tensorboard import TensorboardExperiment

import optuna
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback
import logging
from torch import nn
import torch
from torch.utils.data import random_split
import os

log = logging.getLogger(__name__)
optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.


def train(
    trial: optuna.trial.Trial,
    hparams: dict,
    model: nn.Module,
    cfg: THGStrainStressConfig,
) -> float:
    """Train the model with given trial hyperparameters.

    Args:
        trial: the Optuna trial of the current training
        hparams: the hyperparameters that are used during optimization
        model: the model that is optimized
        cfg: configuration object containing cfg.paths.[data, target]

    Returns:
        loss: the loss defined by the loss function (MAE)
    """

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = model.to(device)
    loss_fn = nn.L1Loss()  # MAE.
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=hparams["lr"],
        betas=(0.9, 0.999),
    )

    # Split the dataset in train, validation and test (sub)sets.
    dataset, groups = THGStrainStressDataset.load_data(
        data_path=cfg.paths.data,
        targets_path=cfg.paths.targets,
    )

    train_test_split = int(len(dataset) * 0.8)
    train_set, test_set = random_split(
        dataset, [train_test_split, len(dataset) - train_test_split]
    )

    train_val_split = int(len(train_set) * 0.8)
    train_subset, val_subset = random_split(
        train_set, [train_val_split, len(train_set) - train_val_split]
    )

    # Define dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=int(hparams["batch_size"]), num_workers=1
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=int(hparams["batch_size"]), num_workers=1
    )

    # Create the runners
    val_runner = Runner(
        loader=val_loader,
        model=model,
        loss_fn=loss_fn,
        stage=Stage.VAL,
        device=device,
        progress_bar=False,
    )
    train_runner = Runner(
        loader=train_loader,
        model=model,
        loss_fn=loss_fn,
        stage=Stage.TRAIN,
        optimizer=optimizer,
        device=device,
        progress_bar=False,
    )

    # Setup the experiment tracker
    log_dir = os.getcwd() + "/tensorboard"
    tracker = TensorboardExperiment(log_path=log_dir)

    # Run epochs.
    for epoch_id in range(500):
        run_epoch(
            val_runner=val_runner,
            train_runner=train_runner,
            experiment=tracker,
            epoch_id=epoch_id,
        )

        loss = val_runner.avg_loss
        trial.report(loss, epoch_id)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return loss


def build_model(trial: optuna.trial.Trial, hparams: dict[str, Any]):
    """Build model within an Optuna trial.

    Args:
        trial: Optuna trial
        hparams: dict of parameters (including those to be optimized by Optuna)

    Returns:
        model: `nn.Sequential` of all parameterized layers
    """

    layers = []

    # Block 1
    layers.append(nn.Conv2d(1, 64, 3))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout2d(hparams["dropout_1"]))
    for _ in range(3):
        layers.append(nn.MaxPool2d(2))

    # Block 2
    layers.append(nn.Conv2d(64, 64, 5))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout2d(hparams["dropout_2"]))
    layers.append(nn.MaxPool2d(2))

    # Block 3
    layers.append(nn.Conv2d(64, 64, 3))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout2d(hparams["dropout_3"]))
    layers.append(nn.MaxPool2d(2))

    # Block 4
    layers.append(nn.Conv2d(64, 64, 6))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout2d(hparams["dropout_4"]))

    # MLP
    layers.append(nn.Flatten())
    layers.append(nn.Linear(64, hparams["n_nodes"]))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(hparams["n_nodes"], hparams["num_output_features"]))

    return nn.Sequential(*layers)


def objective(trial: optuna.trial.Trial, cfg: THGStrainStressConfig):
    """Function for Optuna to optimize.
    Args:
        trial: Optuna trial
        cfg: configuration object

    Returns:
        loss: loss calculated in train()
    """

    # Define hyperparameter space.
    hparams = {
        "lr": trial.suggest_float("lr", *cfg.optuna.lr, log=True),
        "dropout_1": trial.suggest_float("dropout_1", *cfg.optuna.dropout_1),
        "dropout_2": trial.suggest_float("dropout_2", *cfg.optuna.dropout_2),
        "dropout_3": trial.suggest_float("dropout_3", *cfg.optuna.dropout_3),
        "dropout_4": trial.suggest_float("dropout_4", *cfg.optuna.dropout_4),
        "n_nodes": trial.suggest_categorical("n_nodes", cfg.optuna.num_nodes),
        "num_output_features": cfg.model.num_output_features,
        "batch_size": trial.suggest_categorical("batch_size", cfg.optuna.batch_size),
    }

    # Generate the model.
    model = build_model(trial, hparams)

    # Calculate loss and prune trials.
    loss = train(trial, hparams, model, cfg)

    return loss


def tune_hyperparameters(cfg: THGStrainStressConfig):
    """Optimize parameters using an objective function and Optuna.
    All configurations are overwritten during optimization.
    Number of Optuna trials is given by cfg.optuna.trials.
    Optimal hyperparameters are logged.

    Args:
        cfg: hydra configuration object. Only uses cfg.optuna.trials.
    """
    # Create study.
    study_name = cfg.optuna.study_name
    storage = f"sqlite:///{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.optuna.seed),
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=100, reduction_factor=2
        ),
    )

    # Optimize objective in study.
    study.optimize(
        lambda trial: objective(trial, cfg),
        show_progress_bar=True,
        callbacks=[
            MaxTrialsCallback(
                cfg.optuna.trials, states=(TrialState.COMPLETE, TrialState.PRUNED)
            )
        ],
    )

    # Print results.
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    log.info("Study statistics: ")
    log.info("  Number of finished trials: ", len(study.trials))
    log.info("  Number of pruned trials: ", len(pruned_trials))
    log.info("  Number of complete trials: ", len(complete_trials))

    log.info("Best trial:")

    log.info("  Value: ", study.best_trial.value)

    log.info("  Params: ")
    for key, value in study.best_trial.params.items():
        log.info("    {}: {}".format(key, value))
