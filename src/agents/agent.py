import shutil

import numpy as np
import torch
from data.make_dataset import THGStrainStressDataset
from models.models import THGStrainStressCNN
from sklearn.model_selection import KFold
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Resize, ToTensor
from utils.misc import print_cuda_statistics

from agents.base import BaseAgent

cudnn.benchmark = True  # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3


class THGStrainStressAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        # define data_loader
        # self.data_loader = THGStrainStressDataLoader(config)

        # Define dataset
        self.dataset = THGStrainStressDataset(
            config=config,
            labels_file=config.data.labels_file,
            img_dir=config.data.img_dir,
            # THGStrainStressCNN accepts only 258x258 images.
            # Downsample original images with bilinear interpolation.
            transform=Compose([Resize((258, 258)), ToTensor()]),
            label_transform=Compose([]),
        )

        torch.manual_seed(self.config.seed)
        self.check_cuda()

        # initialize counter
        self.current_fold = 1
        self.current_epoch = 1
        self.current_iteration = 0
        self.lowest_loss = np.inf

        # Tensorboard summary writer
        self.initialize_writer()

    def initialize_writer(self):

        charts = {
            f"fold-{fold}": [
                "Multiline",
                [f"fold-{fold}-loss/train", f"fold-{fold}-loss/validation"],
            ]
            for fold in range(1, self.config.k_folds)
        }

        layout = {
            f"THG-STRAIN-STRESS": charts,
        }

        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir)
        self.summary_writer.add_custom_scalars(layout)

    def check_cuda(self):
        cuda_available = torch.cuda.is_available()
        if cuda_available and not self.config.cuda:
            self.logger.info("WARNING: CUDA available. Consider setting cuda to true.")
        elif cuda_available and self.config.cuda:
            torch.cuda.manual_seed_all(self.config.seed)
            self.device = torch.device("cuda:" + str(self.config.gpu_device))
            self.logger.info("Program will run on CUDA-GPU.")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            self.logger.info("Program will run on CPU.\n")

    def initialize_model(self):
        self.model = THGStrainStressCNN(config=self.config).to(self.device)

        # Mean absolute error loss.
        # self.loss = nn.L1Loss(reduction="sum")
        self.loss = nn.L1Loss().to(self.device)

        # Define optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta_1, self.config.beta_2),
        )

    def load_checkpoint(self, filename):
        """Load latest checkpoint.

        Args:
            filename: name of checkpoint file in directory provided by config.checkpoint_dir.
        """

        filename = self.config.checkpoint_dir + filename
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint["epoch"]
            self.current_iteration = checkpoint["iteration"]
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

            self.logger.info(
                "Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n".format(
                    self.config.checkpoint_dir,
                    checkpoint["epoch"],
                    checkpoint["iteration"],
                )
            )
            checkpoint_found = True
        except OSError as e:
            checkpoint_found = False
            self.logger.info(
                "No checkpoint exists from '{}'. Skipping...".format(
                    self.config.checkpoint_dir
                )
            )
            self.logger.info("Training without using checkpoint.")

        return checkpoint_found

    def save_checkpoint(self, filename="checkpoint.pth.tar", is_best=False):
        """Saves checkpoint.

        Args:
            filename: name of checkpoint file in directory provide by config.checkpoint_dir to be loaded.
            is_best: indicate whether current checkpoint's metric is the best so far.
        """
        state = {
            "epoch": self.current_epoch,
            "iteration": self.current_iteration,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        # Save the state
        torch.save(state, self.config.checkpoint_dir + filename)

        # If it is the best, copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(
                self.config.checkpoint_dir + filename,
                self.config.checkpoint_dir + "best-checkpoint.pth.tar",
            )

    def run(self):
        """Main operator."""
        try:
            # Training scheme adapted from
            # https://pytorch-ignite.ai/how-to-guides/07-cross-validation/#training-using-k-fold
            self.k_fold_cross_validation_with_validation_and_test_set()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C. Exiting...")

    def k_fold_cross_validation_with_validation_and_test_set(self):
        """Performs a k-fold cross-validation training scheme
        with a validation and test set.
        """
        self.k_folds = self.config.k_folds

        # Split dataset in test and validation + training set.
        test_set_size = int(len(self.dataset) / self.k_folds)
        train_validation_set_size = len(self.dataset) - test_set_size
        self.test_set, self.train_validation_set = random_split(
            dataset=self.dataset, lengths=[test_set_size, train_validation_set_size]
        )

        # Define split ids.
        splits = KFold(
            n_splits=self.k_folds, shuffle=True, random_state=self.config.seed
        )

        results_per_fold = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            splits.split(np.arange(len(self.train_validation_set)))
        ):
            self.fold = fold_idx + 1
            self.logger.info(f"Training fold {self.fold} has started.")

            (
                train_loader,
                val_loader,
            ) = self.dataset.setup_k_folds_cross_validation_dataflow(
                train_idx=train_idx, val_idx=val_idx
            )
            train_results, val_results = self.train(train_loader, val_loader)
            results_per_fold.append([train_results, val_results])

        test_loader = DataLoader(self.test_set, batch_size=self.config.batch_size_test)
        self.test(test_loader)

    def train(self, train_loader, val_loader):
        """Main training loop."""
        train_results = []
        val_results = []

        self.initialize_model()

        self.current_epoch = 1
        while self.current_epoch <= self.config.max_epoch:
            train_results.append(self.train_one_epoch(train_loader))
            val_results.append(self.validate(val_loader))

            # Make sure all information is flushed from the buffer.
            self.summary_writer.flush()

            self.current_epoch += 1

        return train_results, val_results

    def train_one_epoch(self, train_loader):
        """One epoch of training."""
        # Turn on training mode.
        self.model.train()

        running_training_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device)

            output = self.model(data)
            loss = self.loss(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_training_loss += loss.item()

        avg_loss = running_training_loss / len(train_loader.dataset)

        self.summary_writer.add_scalar(
            f"fold-{self.fold}-loss/train", avg_loss, self.current_epoch
        )

        self.logger.info(
            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                self.current_epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader.dataset),
                loss.item(),
            )
        )

        return avg_loss

    def validate(self, val_loader):
        """One cycle of model validation."""
        # Turn on evaluation mode.
        self.model.eval()

        running_validation_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                running_validation_loss += self.loss(
                    output, target
                )  # sum up batch loss

        avg_loss = running_validation_loss / len(val_loader.dataset)

        self.summary_writer.add_scalar(
            f"fold-{self.fold}-loss/validation", avg_loss, self.current_epoch
        )

        self.logger.info(
            "\nValidation set: Average loss: {:.4f}\n".format(
                avg_loss,
            )
        )

        # Keep track of best performing model.
        if self.lowest_loss > avg_loss:
            self.lowest_loss = avg_loss
            torch.save(
                self.model.state_dict(),
                self.config.checkpoint_dir + "best-checkpoint.pth",
            )
            # self.best_model = self.model
            # self.save_checkpoint()

    def test(self, test_loader):
        """One cycle of model testing.
        TODO: This function is very similar to validate(). Merge them?
        """
        # Load best model.
        self.best_model = THGStrainStressCNN(self.config).to(self.device)
        self.best_model.load_state_dict(
            torch.load(self.config.checkpoint_dir + "best-checkpoint.pth")
        )

        self.best_model.eval()

        running_testing_loss = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.best_model(data)
                running_testing_loss += self.loss(output, target)  # sum up batch loss

        avg_loss = running_testing_loss / len(test_loader.dataset)

        self.logger.info(
            "\nTest set: Average loss: {:.4f}\n".format(
                avg_loss,
            )
        )

    def finalize(self):
        """Finalize all operations of this agent and corresponding dataloader"""
        self.summary_writer.close()
        # self.data_loader.finalize()
