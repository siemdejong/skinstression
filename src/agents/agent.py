import shutil

import numpy as np
import torch
from data.make_dataloader import THGStrainStressDataLoader
from models.models import THGStrainStressCNN
from sklearn.model_selection import KFold
from torch import nn
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.misc import print_cuda_statistics

from agents.base import BaseAgent

cudnn.benchmark = True  # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3


class THGStrainStressAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        # define models
        self.model = THGStrainStressCNN()

        # define data_loader
        self.data_loader = THGStrainStressDataLoader(self.config)

        # Mean absolute error loss.
        # self.loss = nn.L1Loss(reduction="sum")
        self.loss = nn.L1Loss()

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: CUDA available. Consider setting cuda to true.")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed_all(self.manual_seed)
            self.device = torch.device("cuda:" + str(self.config.gpu_device))
            self.model = self.model.to(self.device)
            self.loss = self.loss.to(self.device)
            self.logger.info("Program will run on CUDA-GPU.")
            print_cuda_statistics()
        else:
            self.logger.info("Program will run on CPU.\n")

        # Define optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta_1, self.config.beta_2),
        )

        # initialize counter
        self.current_epoch = 1
        self.current_iteration = 0
        self.lowest_loss = np.inf

        self.save_checkpoint(filename="init-checkpoint.pth.tar")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Tensorboard summary writer
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir)

    def load_checkpoint(self, filename):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
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
        except OSError as e:
            self.logger.info(
                "No checkpoint exists from '{}'. Skipping...".format(
                    self.config.checkpoint_dir
                )
            )
            self.logger.info("Training without using checkpoint.")

    def save_checkpoint(self, filename="checkpoint.pth.tar", is_best=False):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
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
        """
        The main operator
        :return:
        """
        try:
            self.k_fold_cross_validation_with_validation_and_test_set()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C. Exiting...")

    def k_fold_cross_validation_with_validation_and_test_set(self):

        # Training and validation.
        kfold = KFold(n_splits=self.config.k_folds, shuffle=True)

        for fold, (train_ids, validation_ids) in enumerate(
            kfold.split(self.data_loader.train_validation_set)
        ):
            print(f"Fold {fold + 1}.")

            # Reset the weights
            # Subsequent folds must not train further with a pretrained model.
            self.load_checkpoint("init-checkpoint.pth.tar")

            self.data_loader.train_loader = (
                self.data_loader.get_train_validation_subsampler(train_ids)
            )
            self.data_loader.validation_loader = (
                self.data_loader.get_train_validation_subsampler(validation_ids)
            )

            self.train()
            is_best = self.validate()

            self.save_checkpoint(is_best=is_best)

        # Testing.
        self.data_loader.test_loader = self.data_loader.get_test_dataloader()
        self.load_checkpoint("best-checkpoint.pth.tar")
        self.test()

    def train(self):
        """
        Main training loop
        :return:
        """
        self.current_epoch = 1
        while self.current_epoch <= self.config.max_epoch + 1:
            self.train_one_epoch()
            self.validate()

            self.current_epoch += 1

        # Make sure all information is flushed from the buffer.
        self.summary_writer.flush()

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        # Turn on training mode.
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.data_loader.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)

            output = self.model(data)
            loss = self.loss(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_idx % self.config.log_interval == 0:
                self.summary_writer.add_scalar(
                    "Training loss", loss, self.current_epoch
                )

                self.logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        self.current_epoch,
                        batch_idx * len(data),
                        len(self.data_loader.train_loader.dataset),
                        100.0 * batch_idx / len(self.data_loader.train_loader),
                        loss.item(),
                    )
                )
            self.current_iteration += 1

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        # Turn on evaluation mode.
        self.model.eval()

        validation_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.data_loader.validation_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                validation_loss += self.loss(output, target).item()  # sum up batch loss
                # pred = output.max(1, keepdim=True)[
                #     1
                # ]  # get the index of the max log-probability
                # correct += pred.eq(target.view_as(pred)).sum().item()

        validation_loss /= len(self.data_loader.validation_loader.dataset)
        # accuracy = 100.0 * correct / len(self.data_loader.validation_loader.dataset)

        self.logger.info(
            # "\nValidatoin set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            "\nValidation set: Average loss: {:.4f}\n".format(
                validation_loss,
                # correct,
                # len(self.data_loader.validation_loader.dataset),
                # accuracy,
            )
        )

        # Keep track of best performing model.
        is_best = False
        if self.lowest_loss > validation_loss:
            self.lowest_loss = validation_loss
            self.best_model = self.model
            is_best = True
        return is_best

    def test(self):
        self.model.eval()

        validation_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.data_loader.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                print(output, target)
                validation_loss += self.loss(output, target).item()  # sum up batch loss
                # pred = output.max(1, keepdim=True)[
                #     1
                # ]  # get the index of the max log-probability
                # correct += pred.eq(target.view_as(pred)).sum().item()

        validation_loss /= len(self.data_loader.test_loader.dataset)
        # accuracy = 100.0 * correct / len(self.data_loader.test_loader.dataset)

        self.logger.info(
            # "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            "\nTest set: Average loss: {:.4f}\n".format(
                validation_loss,
                # correct,
                # len(self.data_loader.test_loader.dataset),
                # accuracy,
            )
        )

        self.save_checkpoint("best-model-checkpoint.pth.tar")

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.summary_writer.close()
        self.data_loader.finalize()
