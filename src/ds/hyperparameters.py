from typing import Any
from conf.config import THGStrainStressConfig
from ds.dataset import THGStrainStressDataset
from ds.runner import Runner, Stage, run_epoch
from ds.tensorboard import TensorboardExperiment

from torch.optim.lr_scheduler import (
    ChainedScheduler,
    LinearLR,
    CosineAnnealingWarmRestarts,
)

import optuna
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback
import logging
from torch import nn
from torch.utils.data import Subset
import torch
from sklearn.model_selection import train_test_split
import os

log = logging.getLogger(__name__)
optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.


def build_model(
    trial: optuna.trial.Trial, hparams: dict[str, Any], cfg: THGStrainStressConfig
):
    """Build model within an Optuna trial.

    Args:
        trial: Optuna trial
        hparams: dict of parameters (including those to be optimized by Optuna)
        cfg: configuration object

    Returns:
        model: `nn.Sequential` of all parameterized layers
    """

    layers = []

    # Pre-blocks
    preblock_kernel = {7: 106, 14: 53, 53: 14}.get(hparams["num_preblocks"])

    for _ in range(hparams["num_preblocks"]):
        layers.append(nn.Conv2d(1, 1, preblock_kernel))
        layers.append(nn.ReLU())

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
    layers.append(nn.Linear(hparams["n_nodes"], cfg.params.model.num_output_features))

    return nn.Sequential(*layers)


class Objective:
    """Provides the callable Optuna objective.
    Data is loaded only once and attached to the objective.
    """

    def __init__(self, cfg: THGStrainStressConfig, data_cls: nn.Module):

        # Load dataset.
        dataset, person_ids = data_cls.load_data(
            split="train",
            data_path=cfg.paths.data,
            targets_path=cfg.paths.targets,
        )

        # Split the dataset in train, validation and test (sub)sets.
        train_size = int(len(dataset) * 0.8)
        train_idx, val_idx = train_test_split(
            dataset,
            train_size=train_size,
            stratify=person_ids,
            shuffle=True,
            random_state=cfg.seed,
        )

        self.train_subset = Subset(dataset, train_idx)
        self.val_subset = Subset(dataset, val_idx)

    def __call__(
        self,
        trial: optuna.trial.Trial,
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

        # Define hyperparameter space.
        hparams = {
            "optimizer_name": trial.suggest_categorical(
                "optimizer_name", cfg.optuna.hparams.optimizer_name
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", *cfg.optuna.hparams.weight_decay, log=True
            ),
            "lr": trial.suggest_float("lr", *cfg.optuna.hparams.lr, log=True),
            "T_mult": trial.suggest_int("T_mult", *cfg.optuna.hparams.T_mult),
            "num_preblocks": trial.suggest_categorical(
                "num_preblocks", cfg.optuna.hparams.num_preblocks
            ),
            "dropout_1": trial.suggest_float(
                "dropout_1", *cfg.optuna.hparams.dropout_1
            ),
            "dropout_2": trial.suggest_float(
                "dropout_2", *cfg.optuna.hparams.dropout_2
            ),
            "dropout_3": trial.suggest_float(
                "dropout_3", *cfg.optuna.hparams.dropout_3
            ),
            "dropout_4": trial.suggest_float(
                "dropout_4", *cfg.optuna.hparams.dropout_4
            ),
            "n_nodes": trial.suggest_categorical("n_nodes", cfg.optuna.hparams.n_nodes),
            "batch_size": trial.suggest_categorical(
                "batch_size", cfg.optuna.hparams.batch_size
            ),
        }

        use_cuda = torch.cuda.is_available()
        log.info(f"CUDA available: {use_cuda}")
        device = torch.device("cuda" if use_cuda else "cpu")

        model = build_model(trial, hparams, cfg)

        model = model.to(device)
        loss_fn = nn.L1Loss()  # MAE.
        optimizer = getattr(torch.optim, hparams["optimizer_name"])(
            model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"]
        )
        warmup_scheduler = LinearLR(
            optimizer=optimizer, start_factor=0.1, end_factor=1, total_iters=10
        )
        restart_scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer, T_0=400, T_mult=hparams["T_mult"]
        )
        scheduler = ChainedScheduler([warmup_scheduler, restart_scheduler])
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

        # Define dataloaders
        train_loader = torch.utils.data.DataLoader(
            self.train_subset,
            batch_size=int(hparams["batch_size"]),
            num_workers=10,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            self.val_subset,
            batch_size=int(hparams["batch_size"]),
            num_workers=10,
            pin_memory=True,
        )

        # Create the runners
        val_runner = Runner(
            loader=val_loader,
            model=model,
            loss_fn=loss_fn,
            stage=Stage.VAL,
            device=device,
            progress_bar=False,
            scaler=scaler,
        )
        train_runner = Runner(
            loader=train_loader,
            model=model,
            loss_fn=loss_fn,
            stage=Stage.TRAIN,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            progress_bar=False,
            scaler=scaler,
        )

        # Setup the experiment tracker
        log_dir = os.getcwd() + "/tensorboard"
        tracker = TensorboardExperiment(log_path=log_dir)

        # Run epochs.
        for epoch_id in range(cfg.params.epoch_count):
            run_epoch(
                val_runner=val_runner,
                train_runner=train_runner,
                experiment=tracker,
                epoch_id=epoch_id,
            )

            loss = val_runner.avg_loss
            log.info(f"epoch: {epoch_id} | loss: {loss}")

            trial.report(loss, epoch_id)

            if trial.should_prune():
                tracker.add_hparams(hparams)
                raise optuna.exceptions.TrialPruned()

        tracker.add_hparams(hparams, loss)
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
    study = optuna.create_study(
        study_name=cfg.optuna.study_name,
        storage=f"sqlite:///{cfg.paths.optuna_db}",
        sampler=optuna.samplers.TPESampler(seed=cfg.optuna.seed),
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=cfg.optuna.pruner.min_resource,
            reduction_factor=cfg.optuna.pruner.reduction_factor,
        ),
        load_if_exists=True,
    )

    objective = Objective(cfg=cfg, data_cls=THGStrainStressDataset)

    # Optimize objective in study.
    study.optimize(
        lambda trial: objective(trial, cfg),
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
