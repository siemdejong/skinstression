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


def train_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn,
    optimizer: Optimizer,
):
    """Trains a model on a dataset with a given optimizer and loss function.

    Arguments:
        dataloader(DataLoader): training and testing data. Must be compatible with 'model'.
        model (nn.Module): the model to train.
        loss_fn (torch.nn loss function): loss function.

    Note: optimizer.step() updates the model in-place.
    """
    size = len(dataloader.dataset)

    for batch, (inputs, targets) in enumerate(dataloader):

        prediction = model(inputs)
        loss = loss_fn(prediction, targets)

        # Backpropagation.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(inputs)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader: DataLoader, model: nn.Module, loss_fn):
    """Tests a (trained) model on a dataset with a given optimizer and loss function.

    Arguments:
        dataloader(DataLoader): training and testing data. Must be compatible with 'model'.
        model (nn.Module): the model to train.
        loss_fn (torch.nn loss function): loss function.

    Returns:
        accuracy (float): the accuracy of the model (correct # of predictions / dataset size).
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            test_loss += loss_fn(predicted, targets).item()

            # Calculate correct and total. Needed for accuracy calculation.
            correct += (predicted == targets).sum().item()

    avg_loss = test_loss / num_batches
    accuracy = 100 * correct / size

    print("Results:")
    print(f"\tAccuracy:{accuracy:>.4f}")
    print(f"\tAverage loss:{avg_loss:>.4f}")

    return accuracy


def k_fold_cross_validation(
    dataset: Dataset,
    model_fn: nn.Module,
    loss_fn,
    k_folds: int = 5,
    num_epochs: int = 1,
    seed: bool = True,
    optimizer_fn: Union[Optimizer, None] = None,
    learning_rate: float = 1e-4,
    # scheduler_fn: _LRScheduler | ReduceLROnPlateau = None,
) -> nn.Module:
    """Performs a k-fold cross-validation training scheme.

    Arguments:
        dataset (Dataset): training and testing data. Must be compatible with 'model'. TODO: Make inherited Dataclass in data/make_dataset.py
        model_fn (nn.Module): the model to train.
        loss_fn (torch.nn loss function): loss function. TODO: What loss do we need?
        k_folds (int): number of partitions (folds) to divide dataset in.
        num_epochs (int): number of times a fold is used during training.
        seed (bool): perform random subsampling training with fixed seed.
        learning_rate (float): learning rate.
        optimizer_fn (Optimizer | None): the optimizer called during training. Default if None: torch.optim.Adam.

        TODO:
        scheduler_fn (_LRScheduler | ReduceLROnPlateau | None): the learning rate scheduler to be used. Modifies `learning_rate` on the fly.

    Returns:
        results (dict[int]): accuracy per fold.
    """

    # To access fold accuracy results with 'results[fold]'.
    results = {}

    # Set fixed random number seed, so we can compare different runs.
    if seed:
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

    # Define k-fold cross-validator.
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # k-fold cross-validation evaluation.
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

        print(f"Fold {fold}.")

        model = model_fn()

        if optimizer_fn is None:
            optimizer_fn = torch.optim.Adam
        optimizer = optimizer_fn(model.parameters(), lr=learning_rate)

        # Create train- and testloaders using random subsampling.
        trainloader = create_data_loader(dataset, train_ids)
        testloader = create_data_loader(dataset, test_ids)

        for epoch in range(num_epochs):

            print(f"Epoch {epoch} of {num_epochs}")

            train_loop(trainloader, model, loss_fn, optimizer)
            accuracy = test_loop(testloader, model, loss_fn)

        # Save accuracy of last epoch (not best epoch) in results dictionary.
        results[fold] = accuracy

    # Summarize k-fold cross-validation results.
    print(f"k-fold cross-valiation accuracy for {k_folds} folds:")
    for key, value in results.items():
        print(f"\tAccuracy fold {key}: {value:>.4f} %")

    average_accuracy = sum(results.values()) / len(results)
    print(f"Average accuracy: {average_accuracy:>.4f} %")

    return average_accuracy


if __name__ == "__main__":
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    k_fold_cross_validation(
        dataset=None,
        model_fn=VisCNN,
        loss_fn=torch.nn.CrossEntropyLoss(),
        k_folds=5,
        num_epochs=1,
        seed=True,
        optimizer_fn=torch.optim.Adam,
        learning_rate=1e-4,
    )
