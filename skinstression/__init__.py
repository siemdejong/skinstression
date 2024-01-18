"""Skinstression: skin stretch regression using deep learning
	Copyright (C) 2024  Siem de Jong
    See LICENSE for full license.
"""
from skinstression.dataset import SkinstressionDataModule, SkinstressionDataset
from skinstression.model import Skinstression

__all__ = ["Skinstression", "SkinstressionDataset", "SkinstressionDataModule"]

VERSION = "2.0.0"
