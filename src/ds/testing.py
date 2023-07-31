import collections
import logging
import os
from glob import glob
from pathlib import Path

from hydra import utils
import numpy as np
import scipy.stats
from sklearn.model_selection import train_test_split
import torch
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset, DataLoader
from torch.multiprocessing import Queue
from PIL import Image

from conf.config import SkinstressionConfig
from ds.models import SkinstressionCNN
from ds.runner import Runner, run_test
from ds import loss as loss_functions
from ds.tracking import Stage
from ds.tensorboard import TensorboardExperiment
from ds.utils import seed_all, ddp_cleanup, ddp_setup
from ds.logging_setup import setup_worker_logging
from ds.dataset import SkinstressionDataset

log = logging.getLogger(__name__)


def load_checkpoint(cfg: SkinstressionConfig, checkpoint_fn: Path):
    # NOTE: consume_prefix_in_state_dict_if_present() should be used
    # if loading a DDP saved checkpoint to non-DDP.
    checkpoint = torch.load(checkpoint_fn)
    new_state_dict = collections.OrderedDict()
    for k, v in checkpoint["model_state_dict"].items():
        name = k.replace(
            "module.", ""
        )  # remove 'module.' prefix, because of bug in checkpoint saving.
        new_state_dict[name] = v

    model = SkinstressionCNN(cfg)
    model.load_state_dict(new_state_dict)

    log.info(f"Model '{checkpoint_fn}' loaded.")

    return model


def mean_confidence_interval(data, confidence=0.95):
    """Return mean and error from data with a confidence interval."""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h


class Tester:
    def __init__(self, dataset_test: SkinstressionDataset, cfg: SkinstressionConfig):
        self.cfg = cfg

        top_k_dir = Path(f"{utils.get_original_cwd()}/data/top-{cfg.params.top_k}-idx")
        top_k_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Assuming the script is run from the project directory.
            # self.test_idx = np.load(top_k_dir / "train_idx.npy")
            # self.test_idx = np.load(top_k_dir / "new_test_idx.npy")
            self.test_idx = np.load(top_k_dir / "act_new_test_idx.npy")
        except OSError:
            raise Exception(f"test indices need to be available in {top_k_dir}.")

        # dataset_val has the same augmentations as the testset
        self.test_subset = Subset(dataset_test, indices=self.test_idx)

    def __call__(
        self,
        global_rank: int,
        local_rank: int,
        model: SkinstressionCNN,
    ):
        model_sync_bathchnorm = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model_sync_bathchnorm.to(local_rank), device_ids=[local_rank])

        loss_fn = getattr(loss_functions, self.cfg.params.loss_fn)

        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.use_amp)

        test_loader = DataLoader(
            self.test_subset,
            batch_size=int(self.cfg.params.batch_size),
            num_workers=self.cfg.dist.num_workers,
            pin_memory=True,
        )
        test_runner = Runner(
            loader=test_loader,
            model=model,
            loss_fn=loss_fn,
            stage=Stage.TEST,
            global_rank=global_rank,
            local_rank=local_rank,
            progress_bar=False,
            scaler=scaler,
            dry_run=self.cfg.dry_run,
            cfg=self.cfg,
        )

        # Setup the experiment tracker
        if global_rank == 0:
            log_dir = os.getcwd() + "/tensorboard"
            tracker = TensorboardExperiment(log_path=log_dir)
        else:
            tracker = None

        run_test(test_runner, tracker)


@record
def test(
    global_rank: int,
    local_rank: int,
    world_size: int,
    cfg: SkinstressionConfig,
    log_queue: Queue,
):
    """Test the model.
    outputs
    1. the model output with confidence band.
    """

    # Initialize process group
    ddp_setup(global_rank, world_size)

    # Setup logging.
    setup_worker_logging(global_rank, log_queue, cfg.debug)

    # Set and seed device
    torch.cuda.set_device(local_rank)
    seed_all(cfg.seed)

    log.info(f"Hello from global rank: {global_rank}/{world_size}")

    dataset_test, _ = SkinstressionDataset.load_data(
        split="validation",
        data_path=cfg.paths.data,
        targets_path=cfg.paths.targets,
        top_k=cfg.params.top_k,
        reweight="sqrt_inv",
        importances=np.array(cfg.params.importances),
        lds=False,
        extension=cfg.paths.extension,
    )

    tester = Tester(dataset_test, cfg)
    models = [load_checkpoint(cfg, fn) for fn in glob(f"{cfg.paths.model_zoo}/*.pt")]

    for model in models:
        tester(
            global_rank=global_rank,
            local_rank=local_rank,
            model=model,
        )

    ddp_cleanup()
