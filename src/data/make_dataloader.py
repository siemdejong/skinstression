import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from torchvision.transforms.functional import to_tensor

from data.make_dataset import THGStrainStressDataset


class THGStrainStressDataLoader:
    def __init__(self, config):
        """TODO: Currently is tightly coupled to the k-fold cross-validation with test + validation sets scheme. Make efforts to decouple the two."""
        self.config = config

        self.dataset = THGStrainStressDataset(
            labels_file=self.config.data.labels_file,
            img_dir=self.config.data.img_dir,
            # THGStrainStressCNN accepts only 258x258 images.
            # Downsample original images with bilinear interpolation.
            transform=Compose([Resize((258, 258))]),
            label_transform=Compose([]),
        )

        test_set_size = int(len(self.dataset) * 0.2)
        train_validation_set_size = len(self.dataset) - test_set_size

        # Split dataset in test and validation + training set.
        self.test_set, self.train_validation_set = random_split(
            dataset=self.dataset,
            lengths=[test_set_size, train_validation_set_size],
            generator=torch.Generator().manual_seed(1),
        )

    def get_test_dataloader(self):

        test_loader = DataLoader(
            dataset=self.test_set, batch_size=self.config.batch_size_test
        )
        return test_loader

    def get_train_validation_subsampler(self, ids):

        subsampler = SubsetRandomSampler(ids)
        data_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size_train_validation,
            sampler=subsampler,
        )
        return data_loader

    def finalize(self):
        pass
