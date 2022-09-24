from projects.thg-stress-strain.src.models.train_model import create_data_loader
import torch
from torch import nn
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from torch.optim import Optimizer
from models import VisCNN

# from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from typing import Union

# from omegaconf import DictConfig, OmegaConf
# import hydra


class Validator:
    def __init__(self, dataset: Dataset, model: nn.Module, loss_fn, optimizer: Optimizer, ):
        self.dataset = dataset
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
    
    def validation_procedure(self):
        raise NotImplementedError

    def create_data_loader(self):
        self.dataloader = DataLoader(self.dataset)


    def test_has_dataloader(self):
        try:
            self.dataloader
        except AttributeError:
            attr_error_msg = "This Validator instance has no dataloader. \
                Create one with create_data_loader()."
            raise(attr_error_msg) from None


    def train_loop(self):
        """Trains a model on a dataset with a given optimizer and loss function.

        Assumes object instance has dataloader attribute.
        """
        self.test_has_dataloader()

        size = len(self.dataloader.dataset)
        

        for batch, (inputs, targets) in enumerate(self.dataloader):

            prediction = self.model(inputs)
            loss = self.loss_fn(prediction, targets)

            # Backpropagation.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss, current = loss.item(), batch * len(inputs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(self):
        """Tests a (trained) model on a dataset with a given optimizer and loss function.

        Arguments:
            dataloader(DataLoader): training and testing data. Must be compatible with 'model'.
            model (nn.Module): the model to train.
            loss_fn (torch.nn loss function): loss function.

        Returns:
            accuracy (float): the accuracy of the model (correct # of predictions / dataset size).
        """
        self.test_has_dataloader()
        size = len(self.dataloader.dataset)
        num_batches = len(self.dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for inputs, targets in self.dataloader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                test_loss += self.loss_fn(predicted, targets).item()

                # Calculate correct and total. Needed for accuracy calculation.
                correct += (predicted == targets).sum().item()

        avg_loss = test_loss / num_batches
        accuracy = 100 * correct / size

        print("Results:")
        print(f"\tAccuracy:{accuracy:>.4f}")
        print(f"\tAverage loss:{avg_loss:>.4f}")

        return accuracy


class CrossValidator(Validator):
    def __init__(
        self,
        dataset=None,
        model_fn=VisCNN,
        loss_fn=torch.nn.CrossEntropyLoss(),
        k_folds=5,
        num_epochs=1,
        seed=True,
        optimizer_fn=torch.optim.Adam,
        learning_rate=1e-4,
    ):
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.k_folds = k_folds
        self.num_epochs = num_epochs
        self.seed = seed
        self.optimizer_fn = optimizer_fn
        self.learning_rate = learning_rate
