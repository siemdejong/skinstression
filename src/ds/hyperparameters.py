from typing import Any
from conf.config import THGStrainStressConfig
from ds.dataset import THGStrainStressDataset
from ds.runner import Runner, Stage, run_epoch
from ds.tensorboard import TensorboardExperiment
from ds.loss import weighted_l1_loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    LinearLR,
    CosineAnnealingWarmRestarts,
)
from torch.multiprocessing import Queue
from ds.logging_setup import setup_worker_logging
from torch.utils.data.distributed import DistributedSampler
from ds.utils import ddp_cleanup, ddp_setup, seed_all

import os

import optuna
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback
import logging
from torch import nn
from torch.utils.data import Subset
import torch
from sklearn.model_selection import train_test_split
import os
import numpy as np

optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.


def build_model(hparams: dict[str, Any], cfg: THGStrainStressConfig):
    """Build model within an Optuna trial.

    Args:
        hparams: Dict of parameters (including those to be optimized by Optuna).
        cfg: Configuration object.

    Returns:
        model: A `nn.Sequential` of all parameterized layers.
    """

    layers = []

    # Pre-blocks
    preblock_kernel = {7: 106, 14: 53, 53: 14}.get(hparams["num_preblocks"])

    # Pre-blocks
    for _ in range(hparams["num_preblocks"]):
        layers.append(nn.Conv2d(1, 1, preblock_kernel, bias=False))
        layers.append(nn.BatchNorm2d(1))
        layers.append(nn.ReLU())

    # Block 1
    layers.append(nn.Conv2d(1, 64, 3, bias=False))
    layers.append(nn.BatchNorm2d(64))
    layers.append(nn.ReLU())
    # layers.append(nn.Dropout2d(cfg.params.model.dropout_1))
    for _ in range(3):
        layers.append(nn.MaxPool2d(2))

    # Block 2
    layers.append(nn.Conv2d(64, 64, 5, bias=False))
    layers.append(nn.BatchNorm2d(64))
    layers.append(nn.ReLU())
    # layers.append(nn.Dropout2d(cfg.params.model.dropout_2))
    layers.append(nn.MaxPool2d(2))

    # Block 3
    layers.append(nn.Conv2d(64, 64, 3, bias=False))
    layers.append(nn.BatchNorm2d(64))
    layers.append(nn.ReLU())
    # layers.append(nn.Dropout2d(cfg.params.model.dropout_3))
    layers.append(nn.MaxPool2d(2))

    # Block 4
    layers.append(nn.Conv2d(64, 64, 6, bias=False))
    layers.append(nn.BatchNorm2d(64))
    layers.append(nn.ReLU())
    # layers.append(nn.Dropout2d(cfg.params.model.dropout_4))

    # MLP
    layers.append(nn.Flatten())
    layers.append(nn.Linear(64, cfg.params.model.n_nodes, bias=False))
    layers.append(nn.BatchNorm1d(cfg.params.model.n_nodes))
    layers.append(nn.ReLU())
    layers.append(
        nn.Linear(cfg.params.model.n_nodes, cfg.params.model.num_output_features)
    )

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
            reweight="sqrt_inv",
            lds=True,
        )

        # Split the dataset in train, validation and test (sub)sets.
        train_size = int(len(dataset) * 0.8)
        train_idx, val_idx = train_test_split(
            np.arange(len(dataset)),
            train_size=train_size,
            stratify=person_ids,
            shuffle=True,
            random_state=cfg.seed,
        )

        self.train_subset = Subset(dataset, train_idx)
        self.val_subset = Subset(dataset, val_idx)

    def __call__(
        self,
        single_trial: optuna.trial.Trial,
        cfg: THGStrainStressConfig,
        local_rank: int,
        world_size: int,
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

        trial = optuna.integration.TorchDistributedTrial(single_trial, local_rank)

        # Define hyperparameter space.
        hparams = {
            "optimizer_name": trial.suggest_categorical(
                "optimizer_name", cfg.optuna.hparams.optimizer_name
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", *cfg.optuna.hparams.weight_decay, log=True
            ),
            "lr": trial.suggest_float("lr", *cfg.optuna.hparams.lr, log=True),
            "T_0": trial.suggest_int("T_0", *cfg.optuna.hparams.T_0),
            "T_mult": trial.suggest_int("T_mult", *cfg.optuna.hparams.T_mult),
            "num_preblocks": trial.suggest_categorical(
                "num_preblocks", cfg.optuna.hparams.num_preblocks
            ),
            "n_nodes": trial.suggest_categorical("n_nodes", cfg.optuna.hparams.n_nodes),
            "batch_size": trial.suggest_categorical(
                "batch_size", cfg.optuna.hparams.batch_size
            ),
        }

        model = build_model(hparams, cfg)
        model_sync_bathchnorm = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model_sync_bathchnorm.to(local_rank), device_ids=[local_rank])

        loss_fn = weighted_l1_loss  # MAE.
        optimizer = getattr(torch.optim, hparams["optimizer_name"])(
            model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"]
        )
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

        warmup_scheduler = LinearLR(
            optimizer=optimizer, start_factor=0.1, end_factor=1, total_iters=10
        )
        restart_scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=cfg.params.optimizer.T_0,
            T_mult=cfg.params.optimizer.T_mult,
        )
        scheduler = ChainedScheduler([warmup_scheduler, restart_scheduler])

        train_sampler = DistributedSampler(self.train_subset)
        val_sampler = DistributedSampler(self.val_subset)

        # Define dataloaders
        train_loader = torch.utils.data.DataLoader(
            self.train_subset,
            batch_size=int(hparams["batch_size"]),
            # num_workers=cfg.dist.cpus_per_gpu,
            num_workers=int(os.environ.get("SLURM_CPUS_ON_NODE")) // world_size,
            persistent_workers=True,
            pin_memory=True,
            shuffle=False,  # The distributed sampler shuffles for us.
            sampler=train_sampler,
        )
        val_loader = torch.utils.data.DataLoader(
            self.val_subset,
            batch_size=int(hparams["batch_size"]),
            # num_workers=cfg.dist.cpus_per_gpu,
            num_workers=int(os.environ.get("SLURM_CPUS_ON_NODE")) // world_size,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
            sampler=val_sampler,
        )

        # Create the runners
        val_runner = Runner(
            loader=val_loader,
            model=model,
            loss_fn=loss_fn,
            stage=Stage.VAL,
            local_rank=local_rank,
            progress_bar=False,
            scaler=scaler,
            dry_run=cfg.dry_run,
        )
        train_runner = Runner(
            loader=train_loader,
            model=model,
            loss_fn=loss_fn,
            stage=Stage.TRAIN,
            optimizer=optimizer,
            scheduler=scheduler,
            local_rank=local_rank,
            progress_bar=False,
            scaler=scaler,
            dry_run=cfg.dry_run,
        )

        # Setup the experiment tracker
        log_dir = os.getcwd() + "/tensorboard"
        tracker = TensorboardExperiment(log_path=log_dir)

        # Run epochs.
        max_epoch = cfg.params.epoch_count  # if not cfg.dry_run else 1
        for epoch_id in range(max_epoch):
            train_runner.loader.sampler.set_epoch(epoch_id)
            val_runner.loader.sampler.set_epoch(epoch_id)
            run_epoch(
                val_runner=val_runner,
                train_runner=train_runner,
                experiment=tracker,
                epoch_id=epoch_id,
                local_rank=local_rank,
            )

            loss = val_runner.avg_loss
            trial.report(loss, epoch_id)
            logging.info(f"Optuna received the losses!")

            if local_rank == 0:
                logging.info(f"epoch: {epoch_id} | loss: {loss}")

            if trial.should_prune():
                # tracker.add_hparams(hparams)
                raise optuna.exceptions.TrialPruned()

        # tracker.add_hparams(hparams, loss)
        return loss


def tune_hyperparameters(
    local_rank: int, world_size: int, cfg: THGStrainStressConfig, log_queue: Queue
):
    """Optimize parameters using an objective function and Optuna.
    All configurations are overwritten during optimization.
    Number of Optuna trials is given by cfg.optuna.trials.
    Optimal hyperparameters are logged.

    Args:
        cfg: hydra configuration object. Only uses cfg.optuna.trials.
        rank: device rank.
    """

    # Setup logging.
    setup_worker_logging(local_rank, log_queue, cfg.debug)
    logging.info("Test worker log")
    logging.error("Test worker error log")
    logging.debug("Test worker debug log")

    # Set and seed device
    torch.cuda.set_device(local_rank)
    logging.info(f"Hello from device {local_rank} of {world_size}")
    seed_all(cfg.seed)

    # Initialize process group
    ddp_setup(local_rank, world_size)

    objective = Objective(cfg=cfg, data_cls=THGStrainStressDataset)

    maxtrials = MaxTrialsCallback(
        cfg.optuna.trials, states=(TrialState.COMPLETE, TrialState.PRUNED)
    )

    study = None
    if local_rank == 0:
        # Create study.
        study = optuna.create_study(
            study_name=cfg.optuna.study_name,
            storage=f"sqlite:///{cfg.paths.optuna_db}",
            direction=cfg.optuna.direction,
            sampler=optuna.samplers.TPESampler(seed=cfg.optuna.seed),
            pruner=optuna.pruners.SuccessiveHalvingPruner(
                min_resource=cfg.optuna.pruner.min_resource,
                reduction_factor=cfg.optuna.pruner.reduction_factor,
            ),
            load_if_exists=True,
        )

        # Optimize objective in study.
        study.optimize(
            lambda trial: objective(trial, cfg, local_rank, world_size),
            callbacks=[maxtrials],
        )
    else:
        for _ in range(cfg.optuna.trials):
            try:
                objective(None, cfg, local_rank, world_size)
                maxtrials()
            except optuna.TrialPruned:
                pass

    if local_rank == 0:
        assert study is not None
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        logging.info("Study statistics: ")
        logging.info(f"  Number of finished trials: {len(study.trials)}")
        logging.info(f"  Number of pruned trials: {len(pruned_trials)}")
        logging.info(f"  Number of complete trials: {len(complete_trials)}")

        logging.info("Best trial:")
        logging.info(f"  Value: {study.best_trial.value}")
        logging.info("  Params: ")
        for key, value in study.best_trial.params.items():
            logging.info("    {}: {}".format(key, value))

    ddp_cleanup()
