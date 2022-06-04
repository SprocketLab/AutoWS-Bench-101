from pathlib import Path
from typing import Optional, Union
import gzip
import pickle
import numpy as np
from torch import Generator
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms
import requests
import tarfile

from .gendata import gener_spherical_mnist
import torch.utils.data as data_utils
import torch
import os, random

from .dataset import FWRENCHDataset
from .ecg_dataset import ECGDataModule
from .ember import create_vectorized_features, read_vectorized_features, create_metadata


class TorchVisionDataset(FWRENCHDataset):

    DEFAULT_TORCHVISION_DOWNLOAD_PATH = (
        FWRENCHDataset.DATASET_PATH / "torchvision_untransformed"
    )

    def __init__(
        self,
        name: str,
        split: str,
        path: Optional[Union[Path, str]] = None,
        download: bool = True,
        download_path: Optional[Union[Path, str]] = None,
        train_p: float = 0.95,
    ):
        if download_path is None:
            download_path = TorchVisionDataset.DEFAULT_TORCHVISION_DOWNLOAD_PATH

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

    @staticmethod
    def _split(train_valid: TorchDataset, train_p: float):
        train_size = int(len(train_valid) * train_p)
        valid_size = len(train_valid) - train_size

        train_subset, valid_subset = random_split(
            train_valid,
            [train_size, valid_size],
            generator=Generator().manual_seed(FWRENCHDataset.SEED),
        )
        return train_subset, valid_subset

    def transform(self):
        self.write_meta()
        self.write_split()


class MNISTDataset(TorchVisionDataset):
    def __init__(self, split: str, name: str = "MNIST", **kwargs):
        super().__init__(name, split, **kwargs)

    def download(self):
        trainvalid = datasets.MNIST(self.download_path, train=True, download=True)
        train_split, valid_split = TorchVisionDataset._split(
            trainvalid, train_p=self.train_p
        )
        test_split = datasets.MNIST(self.download_path, train=False, download=True)

        valid_size = len(valid_split)
        self._set_path_suffix(valid_size)
        self._set_data(train_split, valid_split, test_split)


class FashionMNISTDataset(TorchVisionDataset):
    def __init__(self, split: str, name: str = "FashionMNIST", **kwargs):
        super().__init__(name, split, **kwargs)

    def download(self):
        trainvalid = datasets.FashionMNIST(
            self.download_path, train=True, download=True
        )
        train_split, valid_split = TorchVisionDataset._split(
            trainvalid, train_p=self.train_p
        )
        test_split = datasets.FashionMNIST(
            self.download_path, train=False, download=True
        )

        valid_size = len(valid_split)
        self._set_path_suffix(valid_size)
        self._set_data(train_split, valid_split, test_split)


class EMNISTDataset(TorchVisionDataset):
    def __init__(self, emnist_split: str, split: str, name: str = "EMNIST", **kwargs):
        self.emnist_split = emnist_split
        super().__init__(name, split, **kwargs)

    def download(self):
        self._set_path_suffix(self.emnist_split)

        trainvalid = datasets.EMNIST(
            self.download_path, split=self.emnist_split, train=True, download=True
        )
        train_split, valid_split = TorchVisionDataset._split(
            trainvalid, train_p=self.train_p
        )
        test_split = datasets.EMNIST(
            self.download_path, split=self.emnist_split, train=False, download=True
        )

        valid_size = len(valid_split)
        self._set_path_suffix(valid_size)
        self._set_data(train_split, valid_split, test_split)


class KMNISTDataset(TorchVisionDataset):
    def __init__(self, split: str, name: str = "KMNIST", **kwargs):
        # print("CONSTRUCTOR")
        super().__init__(name, split, **kwargs)

    def download(self):
        # print("DOWNLOAD PATH", self.download_path)
        # exit()
        trainvalid = datasets.KMNIST(self.download_path, train=True, download=True)
        train_split, valid_split = TorchVisionDataset._split(
            trainvalid, train_p=self.train_p
        )
        test_split = datasets.KMNIST(self.download_path, train=False, download=True)

        valid_size = len(valid_split)
        self._set_path_suffix(valid_size)
        self._set_data(train_split, valid_split, test_split)


class QMNISTDataset(TorchVisionDataset):
    def __init__(self, split: str, name: str = "QMNIST", **kwargs):
        super().__init__(name, split, **kwargs)

    def download(self):
        trainvalid = datasets.QMNIST(self.download_path, train=True, download=True)
        train_split, valid_split = TorchVisionDataset._split(
            trainvalid, train_p=self.train_p
        )
        test_split = datasets.QMNIST(self.download_path, train=False, download=True)

        valid_size = len(valid_split)
        self._set_path_suffix(valid_size)
        self._set_data(train_split, valid_split, test_split)


class SVHNDataset(TorchVisionDataset):
    def __init__(self, split: str, name: str = "SVHN", **kwargs):
        super().__init__(name, split, **kwargs)

    def download(self):
        trainvalid = datasets.SVHN(self.download_path, split="train", download=True)
        train_split, valid_split = TorchVisionDataset._split(
            trainvalid, train_p=self.train_p
        )
        test_split = datasets.SVHN(self.download_path, split="test", download=True)

        valid_size = len(valid_split)
        self._set_path_suffix(valid_size)
        self._set_data(train_split, valid_split, test_split)


class CIFAR10Dataset(TorchVisionDataset):
    def __init__(self, split: str, name: str = "CIFAR10", **kwargs):
        super().__init__(name, split, **kwargs)

    def download(self):
        trans = transforms.ToTensor()
        trainvalid = datasets.CIFAR10(
            self.download_path, train=True, transform=trans, download=True
        )
        train_split, valid_split = TorchVisionDataset._split(
            trainvalid, train_p=self.train_p
        )
        test_split = datasets.CIFAR10(
            self.download_path, train=False, transform=trans, download=True
        )
        valid_size = len(valid_split)
        self._set_path_suffix(valid_size)
        self._set_data(train_split, valid_split, test_split)


class CIFAR100Dataset(TorchVisionDataset):
    def __init__(self, split: str, name: str = "CIFAR100", **kwargs):
        super().__init__(name, split, **kwargs)

    def download(self):
        trainvalid = datasets.CIFAR100(self.download_path, train=True, download=True)
        train_split, valid_split = TorchVisionDataset._split(
            trainvalid, train_p=self.train_p
        )
        test_split = datasets.CIFAR100(self.download_path, train=False, download=True)

        valid_size = len(valid_split)
        self._set_path_suffix(valid_size)
        self._set_data(train_split, valid_split, test_split)


class SphericalDataset(TorchVisionDataset):
    def __init__(self, split: str, name: str = "SPHERICAL_MNIST", **kwargs):
        gener_spherical_mnist()
        with gzip.open("s2_mnist.gz", "rb") as f:
            self.genDataset = pickle.load(f)
        super().__init__(name, split, **kwargs)

    def download(self):
        train_data = torch.from_numpy(
            self.genDataset["train"]["images"][:, None, :, :].astype(np.float32)
        )
        train_labels = torch.from_numpy(
            self.genDataset["train"]["labels"].astype(np.int64)
        )
        trainvalid = data_utils.TensorDataset(train_data, train_labels)
        train_split, valid_split = TorchVisionDataset._split(
            trainvalid, train_p=self.train_p
        )
        test_data = torch.from_numpy(
            self.genDataset["test"]["images"][:, None, :, :].astype(np.float32)
        )
        test_labels = torch.from_numpy(
            self.genDataset["test"]["labels"].astype(np.int64)
        )

        test_split = data_utils.TensorDataset(test_data, test_labels)
        valid_size = len(valid_split)
        self._set_path_suffix(valid_size)
        self._set_data(train_split, valid_split, test_split)


class ECG_Time_Series_Dataset(TorchVisionDataset):
    def __init__(self, split: str, name: str = "ECG", **kwargs):
        self.data_module = ECGDataModule()
        self.data_module.setup(stage=None)
        super().__init__(name, split, **kwargs)

    def download(self):

        train_data = self.data_module.ecg_train.data
        train_labels = self.data_module.ecg_train.targets

        valid_data = self.data_module.ecg_valid.data
        valid_labels = self.data_module.ecg_valid.targets

        trainvalid_data = torch.from_numpy(
            np.concatenate((train_data, valid_data), axis=0)
        )
        trainvalid_labels = torch.from_numpy(
            (np.concatenate((train_labels, valid_labels), axis=0))
        )

        trainvalid = data_utils.TensorDataset(trainvalid_data, trainvalid_labels)

        train_split, valid_split = TorchVisionDataset._split(
            trainvalid, train_p=self.train_p
        )

        test_data = torch.from_numpy(self.data_module.ecg_test.data)
        test_labels = torch.from_numpy(self.data_module.ecg_test.targets)

        test_split = data_utils.TensorDataset(test_data, test_labels)
        valid_size = len(valid_split)

        self._set_path_suffix(valid_size)

        if self.split == "train":
            self._set_data(train_split, None, None)
        elif self.split == "valid":
            self._set_data(None, valid_split, None)
        elif self.split == "test":
            self._set_data(None, None, test_split)


class EmberDataset(TorchVisionDataset):
    def __init__(self, split: str, name: str = "ember_2017", **kwargs):
        url = "https://ember.elastic.co/ember_dataset_2017_2.tar.bz2"
        if not (os.path.exists("ember_dataset_2017_2.tar.bz2")):
            print("ADD NEW DATASET")
            target_path = "ember_dataset_2017_2.tar.bz2"
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(target_path, "wb") as f:
                    f.write(response.raw.read())
        if not (os.path.isdir('ember_2017_2')):
            tar = tarfile.open("ember_dataset_2017_2.tar.bz2", "r:bz2")  
            tar.extractall()
            tar.close()
        super().__init__(name, split, **kwargs)

    def download(self):
        data_dir = "ember_2017_2"
        create_vectorized_features(data_dir)
        #_ = create_metadata(data_dir)
        X_train, y_train = read_vectorized_features(data_dir, "train")
        # Filter unlabeled data
        train_rows = (y_train != -1)
        X_train = X_train[train_rows]
        y_train = y_train[train_rows]
        train_selection = random.sample(range(X_train.shape[0]), int(X_train.shape[0]*0.5))
        train_data = torch.from_numpy(X_train[train_selection])
        train_labels = torch.from_numpy(y_train[train_selection].astype(int))
        trainvalid = data_utils.TensorDataset(train_data, train_labels)
        train_split, valid_split = TorchVisionDataset._split(
            trainvalid, train_p=self.train_p
        )
        X_test, y_test = read_vectorized_features(data_dir, "test")
        test_rows = (y_test != -1)
        X_test = X_test[test_rows]
        y_test = y_test[test_rows]
        test_selection = random.sample(range(X_test.shape[0]), int(X_test.shape[0]*0.5))
        test_data = torch.from_numpy(X_test[test_selection])
        test_labels = torch.from_numpy(y_test[test_selection].astype(int))
        test_split = data_utils.TensorDataset(test_data, test_labels)
        valid_size = len(valid_split)
        self._set_path_suffix(valid_size)
        self._set_data(train_split, valid_split, test_split)


def main():
    # MNISTDataset("train")
    # MNISTDataset("valid")
    # MNISTDataset("test")

    # FashionMNISTDataset("train")
    # FashionMNISTDataset("valid")
    # FashionMNISTDataset("test")

    # EMNISTDataset("byclass", "train")
    # EMNISTDataset("byclass", "valid")
    # EMNISTDataset("byclass", "test")

    # KMNISTDataset("train")
    # KMNISTDataset("valid")
    # KMNISTDataset("test")

    # QMNISTDataset("train")
    # QMNISTDataset("valid")
    # QMNISTDataset("test")

    # SVHNDataset("train")
    # SVHNDataset("valid")
    # SVHNDataset("test")

    # CIFAR10Dataset("train")
    # CIFAR10Dataset("valid")
    # CIFAR10Dataset("test")

    # CIFAR100Dataset("train")
    # CIFAR100Dataset("valid")
    # CIFAR100Dataset("test")

    # ECG_Time_Series_Dataset("train")
    # ECG_Time_Series_Dataset("valid")
    # ECG_Time_Series_Dataset("test")

    SphericalDataset("train")
    SphericalDataset("valid")
    SphericalDataset("test")


if __name__ == "__main__":
    main()
