import torch
from torch import nn
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from torch.optim import Optimizer

torch.manual_seed(0)
torch.cuda.manual_seed(0)

# from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from typing import Union

# from omegaconf import DictConfig, OmegaConf
# import hydra


class KFoldCrossValidator:
    def __init__(self, model_fn: nn.Module, loss_fn, optimizer_fn: Optimizer):
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn

    @staticmethod
    def create_data_loader(dataset: Dataset, ids: list[int]):
        """Create Dataloader for a given dataset using random subsampling.

        Arguments:
            dataset (torch.utils.data.Dataset): given dataset to subsample from. TODO: Make inherited Dataset in data/make_dataset.py
            ids (list[int]): indices of dataset to subsample

        Returns:
            loader (DataLoader): iterable version of dataset.
        """
        # Sample elements randomly from a given list of ids, no replacement.
        subsampler = SubsetRandomSampler(ids)

        # Define data loaders for training and testing data in this fold
        loader = DataLoader(dataset, batch_size=10, sampler=subsampler)

        return loader

    def validation_procedure(
        self,
        dataset: Dataset,
        k_folds: int = 5,
        shuffle: bool = True,
        learning_rate: float = 1e-4,
        num_epochs: int = 1,
    ):
        """Performs a k-fold cross-validation training scheme.

        Arguments:
            dataset (Dataset): training and testing data. Must be compatible with 'model'. TODO: Make inherited Dataclass in data/make_dataset.py
            k_folds (int): number of partitions (folds) to divide dataset in.
            num_epochs (int): number of times a fold is used during training.
            learning_rate (float): learning rate.

            TODO:
            scheduler_fn (_LRScheduler | ReduceLROnPlateau | None): the learning rate scheduler to be used. Modifies `learning_rate` on the fly.

        Returns:
            results (dict[int]): accuracy per fold.
        """
        # To access fold accuracy results with 'results[fold]'.
        results = {}

        # Set fixed random number seed, so we can compare different runs.

        # Define k-fold cross-validator.
        kfold = KFold(n_splits=k_folds, shuffle=shuffle)

        # k-fold cross-validation evaluation.
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

            print(f"Fold {fold}.")

            model = self.model_fn()

            if self.optimizer_fn is None:
                self.optimizer_fn = torch.optim.Adam
            optimizer = self.optimizer_fn(model.parameters(), lr=learning_rate)

            # Create train- and testloaders using random subsampling.
            trainloader = KFoldCrossValidator.create_data_loader(dataset, train_ids)
            testloader = KFoldCrossValidator.create_data_loader(dataset, test_ids)

            for epoch in range(num_epochs):

                print(f"Epoch {epoch} of {num_epochs}")

                self.train_loop(trainloader, optimizer)
                accuracy = self.test_loop(testloader)

            # Save accuracy of last epoch (not best epoch) in results dictionary.
            results[fold] = accuracy

        # Summarize k-fold cross-validation results.
        print(f"k-fold cross-valiation accuracy for {k_folds} folds:")
        for key, value in results.items():
            print(f"\tAccuracy fold {key}: {value:>.4f} %")

        average_accuracy = sum(results.values()) / len(results)
        print(f"Average accuracy: {average_accuracy:>.4f} %")

        return average_accuracy

    def train_loop(self, dataloader: DataLoader):
        """Trains the model attribute on a dataset.

        Arguments:
            dataloader (DataLoader): training data. Must be compatible with 'model'.
        """

        size = len(dataloader.dataset)

        for batch, (inputs, targets) in enumerate(dataloader):

            prediction = self.model(inputs)
            loss = self.loss_fn(prediction, targets)

            # Backpropagation.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss, current = loss.item(), batch * len(inputs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(self, dataloader: DataLoader):
        """Tests a (trained) model on a dataset with a given optimizer and loss function.

        Arguments:
            dataloader (DataLoader): testing data. Must be compatible with 'model'.

        Returns:
            accuracy (float): the accuracy of the model (correct # of predictions / dataset size).
        """
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for inputs, targets in dataloader:
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
