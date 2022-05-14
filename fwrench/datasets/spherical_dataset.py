from pathlib import Path
from typing import Optional, Union
import gzip
import pickle
import numpy as np
import torch.utils.data as data_utils
from torch import Generator
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import torch
import os

from .gendata import gener_spherical_mnist
from .dataset import FWRENCHDataset

MNIST_PATH = "s2_mnist.gz"

class Spherical(FWRENCHDataset):
    def __init__(
        self,
        name: str,
        split: str,
        path: Optional[Union[Path, str]] = None,
        download: bool = True,
        download_path: Optional[Union[Path, str]] = None,
        train_p: float = 0.95,
        ):
        self.train_p = train_p
        super().__init__(name, split, path, download, download_path)

    def _set_path_suffix(self, validation_size: int):
        self.path = self.path.with_name(f"{self.path.name}_{validation_size}")

    def _set_data(
        self,
        train_split: TorchDataset,
        valid_split: TorchDataset,
        test_split: TorchDataset,
    ):
        if self.split == "train":
            self.data = train_split
        elif self.split == "valid":
            self.data = valid_split
        elif self.split == "test":
            self.data = test_split

    def transform(self):
        self.write_meta()
        self.write_split()

    def download(self):
        gener_spherical_mnist()
        with gzip.open(MNIST_PATH, 'rb') as f:
            dataset = pickle.load(f)

        train_data = torch.from_numpy(
            dataset["train"]["images"][:, None, :, :].astype(np.float32))
        train_labels = torch.from_numpy(
            dataset["train"]["labels"].astype(np.int64))
        train_dataset = data_utils.TensorDataset(train_data, train_labels)
        train_size = int(len(train_dataset) * self.train_p)
        valid_size = len(train_dataset) - train_size

        train_subset, valid_subset = random_split(
            train_dataset,
            [train_size, valid_size],
            generator=Generator().manual_seed(FWRENCHDataset.SEED),
        )
        test_data = torch.from_numpy(
            dataset["test"]["images"][:, None, :, :].astype(np.float32))
        test_labels = torch.from_numpy(
            dataset["test"]["labels"].astype(np.int64))

        test_dataset = data_utils.TensorDataset(test_data, test_labels)
        valid_size = len(test_dataset)
        self._set_path_suffix(valid_size)
        self._set_data(train_subset, valid_subset, test_dataset)