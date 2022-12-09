"""Implements a hyperparameter opimization algorithm using Optuna.
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
import os
from typing import Any

from hydra import utils
import numpy as np
import optuna
import torch
from optuna.storages import RetryFailedTrialCallback
from optuna.trial import TrialState
from sklearn.model_selection import train_test_split
from torch import nn
from torch.distributed.elastic.multiprocessing.errors import record
from torch.multiprocessing import Queue
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import (
    SequentialLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
)
from torch.utils.data import Subset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from conf.config import SkinstressionConfig
from ds.dataset import SkinstressionDataset
from ds.logging_setup import setup_worker_logging
from ds import loss as loss_functions
from ds.runner import Runner, Stage, run_epoch
from ds.tensorboard import TensorboardExperiment
from ds.utils import ddp_cleanup, ddp_setup, seed_all
from ds.exceptions import IOErrorAfterRetries

optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

log = logging.getLogger(__name__)


def build_model(hparams: dict[str, Any], cfg: SkinstressionConfig):
    """Build model within an Optuna trial.

    Args:
        hparams: Dict of parameters (including those to be optimized by Optuna).
        cfg: Configuration object.

    Returns:
        model: A `nn.Sequential` of all parameterized layers.
    """

    layers = []

    # Pre-blocks
    # preblock_kernel = {7: 106, 14: 53, 53: 14}.get(hparams["num_preblocks"])

    # Pre-blocks
    # for _ in range(hparams["num_preblocks"]):
    #     layers.append(nn.Conv2d(1, 1, preblock_kernel, bias=False))
    #     layers.append(nn.BatchNorm2d(1))
    #     layers.append(nn.ReLU())

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

    def __init__(self, cfg: SkinstressionConfig, data_cls: nn.Module):
        """Initialize the objective to preload data
        and share train/validation splits across trials.

        Args:
            cfg: configuration object containing cfg.paths.[data, target]
                 and optuna hyperparameter space.
            data_cls: Pytorch Dataset class with a load_data method to obtain
                      the full dataset and ids to stratify on.
        """
        self.cfg = cfg

        # TODO: Below is a dirty copy from training.py. DRY.
        # Load dataset.
        # class_ids are persons we need to stratify on.
        dataset_train, class_ids = data_cls.load_data(
            split="train",
            data_path=cfg.paths.data,
            targets_path=cfg.paths.targets,
            top_k=cfg.params.top_k,
            reweight="sqrt_inv",
            importances=np.array(cfg.params.importances),
            lds=True,
            extension=cfg.paths.extension,
        )
        dataset_val, _ = data_cls.load_data(
            split="validation",
            data_path=cfg.paths.data,
            targets_path=cfg.paths.targets,
            top_k=cfg.params.top_k,
            reweight="sqrt_inv",
            importances=np.array(cfg.params.importances),
            lds=False,
            extension=cfg.paths.extension,
        )

        try:
            # Assuming the script is run from the project directory.
            train_idx = np.load(f"{utils.get_original_cwd()}/data/train_idx.npy")
            val_idx = np.load(f"{utils.get_original_cwd()}/data/val_idx.npy")
            test_idx = np.load(f"{utils.get_original_cwd()}/data/test_idx.npy")
            train_val_idx = np.concatenate((train_idx, val_idx))
        except OSError:
            # Split the dataset in train, validation and test (sub)sets.
            train_val_size = int(len(dataset_train) * 0.8)
            train_val_idx, test_idx = train_test_split(
                np.arange(len(dataset_train)),
                train_size=train_val_size,
                stratify=class_ids,
                shuffle=True,
                random_state=cfg.seed,
            )

            train_size = int(len(train_val_idx) * 0.8)
            train_idx, val_idx = train_test_split(
                np.arange(len(train_val_idx)),
                train_size=train_size,
                stratify=class_ids[train_val_idx],
                shuffle=True,
                random_state=cfg.seed,
            )

            log.info(f"train idx: {train_idx}")
            log.info(f"val idx: {val_idx}")
            log.info(f"test idx: {test_idx}")

            np.save(f"{utils.get_original_cwd()}/data/train_idx.npy", train_idx)
            np.save(f"{utils.get_original_cwd()}/data/val_idx.npy", val_idx)
            np.save(f"{utils.get_original_cwd()}/data/test_idx.npy", test_idx)

        # TODO: CROSSRUNNERS NOW ONLY USE NON-AUGMENTED DATA!
        self.train_val_subset = Subset(dataset_val, indices=train_val_idx)
        self.train_subset = Subset(dataset_train, indices=train_idx)
        self.val_subset = Subset(dataset_val, indices=val_idx)
        # dataset_val has the same augmentations as the testset
        self.test_subset = Subset(dataset_val, indices=test_idx)

    def __call__(
        self,
        single_trial: optuna.trial.Trial,
        global_rank: int,
        local_rank: int,
        world_size: int,
    ) -> float:
        """Train the model with given trial hyperparameters.

        Args:
            single_trial: the Optuna trial of the current training.
            global_rank: the rank across all nodes.
            local_rank: the rank on the node.
            world_size: the world size of the current process group.

        Returns:
            loss: the optimization loss.
        """

        # Convert trials to a distributed trial so Optuna knows that multiple
        # GPUs in its process group are used per trial.
        # Watch https://github.com/optuna/optuna/pull/4106 for an interesting discussion
        # on the use of the `device` keyword argument.
        trial = optuna.integration.TorchDistributedTrial(single_trial, local_rank)

        # Define hyperparameter space.
        hparams = {
            "weight_decay": trial.suggest_float(
                "weight_decay", *self.cfg.optuna.hparams.weight_decay, log=True
            ),
            "lr": trial.suggest_float("lr", *self.cfg.optuna.hparams.lr, log=True),
            "T_0": trial.suggest_int("T_0", *self.cfg.optuna.hparams.T_0),
            "T_mult": trial.suggest_int("T_mult", *self.cfg.optuna.hparams.T_mult),
            "n_nodes": trial.suggest_int("n_nodes", *self.cfg.optuna.hparams.n_nodes),
            "batch_size": trial.suggest_int(
                "batch_size", *self.cfg.optuna.hparams.batch_size
            ),
        }

        if global_rank == 0:
            log.info(f"Hyperparameters of trial {trial.number}: {hparams}")

        # Build the model with hparams defined by Optuna.
        # Use SyncBatchNorm to sync statistics between parallel models.
        # Cast the model to a DistributedDataParallel model where every GPU
        # has the full model.
        model = build_model(hparams, self.cfg)
        model_sync_bathchnorm = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model_sync_bathchnorm.to(local_rank), device_ids=[local_rank])

        # Choose optimizer with learning rates picked by Optuna.
        loss_fn = getattr(loss_functions, self.cfg.params.loss_fn)
        optimizer = getattr(torch.optim, self.cfg.params.optimizer.name)(
            model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"]
        )

        # Used for automatic mixed precision (AMP) to prevent underflow.
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.use_amp)

        # Stabilize the initial model.
        warmup_scheduler = LinearLR(
            optimizer=optimizer,
            start_factor=0.1,
            end_factor=1,
            total_iters=self.cfg.params.scheduler.T_warmup,
        )

        # Cosine annealing is used as a way to get to local minima.
        # Warm restarts are used to try and find neighbouring local minima.
        # https://arxiv.org/abs/1608.03983
        restart_scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=hparams["T_0"],
            T_mult=hparams["T_mult"],
        )

        scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_scheduler, restart_scheduler],
            milestones=[self.cfg.params.scheduler.T_warmup],
        )

        # Distributed the workload across the GPUs.
        train_sampler = DistributedSampler(self.train_subset, seed=self.cfg.seed)
        val_sampler = DistributedSampler(self.val_subset, seed=self.cfg.seed)

        # Define dataloaders
        train_loader = DataLoader(
            self.train_subset,
            batch_size=int(hparams["batch_size"]),
            num_workers=self.cfg.dist.num_workers,
            persistent_workers=True,
            pin_memory=True,
            shuffle=False,  # The distributed sampler shuffles for us.
            sampler=train_sampler,
        )
        val_loader = DataLoader(
            self.val_subset,
            batch_size=int(hparams["batch_size"]),
            num_workers=self.cfg.dist.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,  # The distributed sampler shuffles for us.
            sampler=val_sampler,
        )

        # Create the runners
        train_runner = Runner(
            loader=train_loader,
            model=model,
            loss_fn=loss_fn,
            stage=Stage.TRAIN,
            optimizer=optimizer,
            scheduler=scheduler,
            global_rank=global_rank,
            local_rank=local_rank,
            progress_bar=self.cfg.progress_bar,
            scaler=scaler,
            dry_run=self.cfg.dry_run,
            trial=trial.number,
        )
        val_runner = Runner(
            loader=val_loader,
            model=model,
            loss_fn=loss_fn,
            stage=Stage.VAL,
            global_rank=global_rank,
            local_rank=local_rank,
            progress_bar=self.cfg.progress_bar,
            scaler=scaler,
            dry_run=self.cfg.dry_run,
            trial=trial.number,
        )

        # Setup the experiment tracker
        if global_rank == 0:
            log_dir = os.getcwd() + "/tensorboard"
            tracker = TensorboardExperiment(log_path=log_dir)
        else:
            tracker = None

        # Run epochs.
        max_epoch = self.cfg.optuna.pruner.max_resource
        for epoch_id in range(max_epoch):
            train_runner.loader.sampler.set_epoch(epoch_id)
            val_runner.loader.sampler.set_epoch(epoch_id)

            run_epoch(
                val_runner=val_runner,
                train_runner=train_runner,
                epoch_id=epoch_id,
                experiment=tracker,
            )

            train_loss = train_runner.avg_loss
            val_loss = val_runner.avg_loss

            if global_rank == 0:
                log.info(f"epoch: {epoch_id} | train loss: {train_loss}")
                log.info(f"epoch: {epoch_id} | validation loss: {val_loss}")

            trial.report(val_loss, epoch_id)

            if trial.should_prune():
                # TODO: compile Pytorch with Caffe2.
                # tracker.add_hparams(hparams)
                raise optuna.exceptions.TrialPruned()

            train_runner.reset()
            val_runner.reset()

        if tracker:
            tracker.flush()

        # tracker.add_hparams(hparams, loss)
        return val_loss


@record
def tune_hyperparameters(
    global_rank: int,
    local_rank: int,
    world_size: int,
    cfg: SkinstressionConfig,
    log_queue: Queue,
):
    """Optimize parameters using an objective function and Optuna.
    All configurations are overwritten during optimization.
    Number of Optuna trials is given by cfg.optuna.trials.
    Optimal hyperparameters are logged.

    Args:
        global_rank: the GPU rank in the world.
        local_rank: the GPU rank on the node.
        world_size: the number of GPUs on the node.
        cfg: configuration object containing cfg.paths.[data, target]
             and optuna hyperparameter space.
        log_queue: a queue for logging to push their messages to.
                   Can be created by `setup_primary_logging()`.
    """

    # Setup logging.
    setup_worker_logging(global_rank, log_queue, cfg.debug)

    # Set and seed device
    torch.cuda.set_device(local_rank)
    log.info(f"Hello from device {global_rank} of {world_size}")
    seed_all(cfg.seed)

    # Initialize process group
    ddp_setup(global_rank, world_size)

    # Load data into objective, so it doesn't need to load every trial.
    objective = Objective(cfg=cfg, data_cls=SkinstressionDataset)

    study = None
    if global_rank == 0:
        # Recording heartbeats every 60 seconds.
        # Other processes' trials where more than 120 seconds have passed
        # since the last heartbeat was recorded will be automatically failed.
        # Retry a failed process.
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{cfg.paths.optuna_db}",
            heartbeat_interval=60,
            grace_period=120,
            failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
        )

        # Create a study in the database.
        # Define study direction.
        # Use the TPE sampler, that samples close to promising trials.
        # Prune with ASHA algorithm.
        # Set min_resource to be something that brings the loss close to 0.
        # Reduction factor can be larger if more trials are allowed.
        sampler_cls = getattr(optuna.samplers, cfg.optuna.sampler.name)

        sampler = None
        if sampler_cls.__name__ == "CmaEsSampler":
            consider_pruned_trials = (
                True
                if cfg.optuna.sampler.name in ["HyperbandPruner", "SuccessiveHalving"]
                else False
            )
            sampler = sampler_cls(
                seed=cfg.optuna.sampler.seed,
                restart_strategy=cfg.optuna.sampler.restart_strategy,
                inc_popsize=cfg.optuna.sampler.inc_popsize,
                consider_pruned_trials=consider_pruned_trials,
            )
        elif sampler_cls.__name__ == "TPESampler":
            sampler = sampler_cls(seed=cfg.optuna.sampler.seed)

        pruner_cls = getattr(optuna.pruners, cfg.optuna.pruner.name)

        if pruner_cls.__name__ == "HyperbandPruner":
            # To make pruning reproducible for Hyperband
            os.environ["PYTHONHASHSEED"] = str(cfg.optuna.pruner.seed)
            pruner = pruner_cls(
                min_resource=cfg.optuna.pruner.min_resource,
                max_resource=cfg.optuna.pruner.max_resource,
                reduction_factor=cfg.optuna.pruner.reduction_factor,
            )
        elif pruner_cls.__name__ == "SuccessiveHalving":
            pruner = pruner_cls(
                min_resource=cfg.optuna.pruner.min_resource,
                reduction_factor=cfg.optuna.pruner.reduction_factor,
            )
        else:
            pruner = None

        study = optuna.create_study(
            study_name=cfg.optuna.study_name,
            storage=storage,
            direction=cfg.optuna.direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        # Optimize objective in study.
        study.optimize(
            lambda trial: objective(trial, global_rank, local_rank, world_size),
            n_trials=cfg.optuna.trials,
            gc_after_trial=True,
            catch=(IOErrorAfterRetries,),
        )
    else:

        # Other nodes work with rank 0 to get the trial done
        # as fast as possible.
        for _ in range(cfg.optuna.trials):
            try:
                objective(None, global_rank, local_rank, world_size)
            except optuna.TrialPruned:
                pass

    if global_rank == 0:
        assert study is not None
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        log.info("Study statistics: ")
        log.info(f"  Number of finished trials: {len(study.trials)}")
        log.info(f"  Number of pruned trials: {len(pruned_trials)}")
        log.info(f"  Number of complete trials: {len(complete_trials)}")

        log.info("Best trial:")
        log.info(f"  Value: {study.best_trial.value}")
        log.info("  Params: ")
        for key, value in study.best_trial.params.items():
            log.info("    {}: {}".format(key, value))

    # Exit subprocesses.
    ddp_cleanup()
