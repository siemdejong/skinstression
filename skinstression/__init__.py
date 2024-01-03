"""Skinstression: skin stretch regression using deep learning
	Copyright (C) 2024  Siem de Jong
    See LICENSE for full license.
"""
from dataset import SkinstressionDataModule, SkinstressionDataset
from model import Skinstression

__all__ = ["Skinstression", "SkinstressionDataset", "SkinstressionDataModule"]

VERSION = "2.0.0"
