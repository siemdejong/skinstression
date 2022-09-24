# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path, PurePath
import numpy
from PIL import Image
from dotenv import find_dotenv, load_dotenv
import os
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
import glob


class THGSkinImagePreprocess:
    """Helper class for everything THG skin image preprocessing related."""

    @staticmethod
    def batch_convert_tif_to_png(source_img_dir, target_img_dir, labels_path=None):
        """Converts e.g. data/interim/1__00_.tif to data/processed/patches/1__00_.png

        Arguments:
            sourc_img_dir (str): directory of the source images.
            target_img_dir (str): directory of the target images.
            labels_path (str): file with labels. If set, converts the label file
                to reflect converted image names. Assumes labels stored in .csv file.
                Defaults to None.
        """

        target_extension = ".png"

        # Convert images.
        search_dir = os.path.join(source_img_dir, "*.tif")
        source_img_paths = glob.glob(search_dir)
        for source_img_path in source_img_paths:
            source_img_filename = os.path.basename(source_img_path)
            target_img_filename = (
                os.path.splitext(source_img_filename)[0] + target_extension
            )
            target_img_path = os.path.join(target_img_dir, target_img_filename)
            with Image.open(source_img_path) as im:
                im.save(target_img_path)

        # Convert image index in labels file.
        labels_df = pd.read_csv(labels_path)
        labels_df["id"] = labels_df["id"].replace(
            {"(.tif)": target_extension}, regex=True
        )
        labels_filename = os.path.basename(labels_path)
        target_labels_path = os.path.join(target_img_dir, labels_filename)
        labels_df.to_csv(target_labels_path, index=False)


class THGSkinImageDataset(Dataset):
    def __init__(self, labels_file, img_dir, transform=None, target_transform=None):
        # header = 0, assume there is a header in the labels.csv file.
        self.img_labels = pd.read_csv(labels_file, header=0)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        labels = self.img_labels.iloc[idx, [1, 2, 3]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        return image, labels


class RandomImageDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


@click.command()
@click.argument("output_filepath", type=click.Path())
def generate_random_images(output_filepath):
    """Generate random images (saved in ../mock)

    Inspired by https://stackoverflow.com/a/10901092.
    """

    for n in range(50):
        a = numpy.random.rand(258, 258) * 255
        im_out = Image.fromarray(a.astype("uint8"))
        im_out.save(os.path.join(output_filepath, "out%000d.jpg" % n))


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    # main()

    generate_random_images()
