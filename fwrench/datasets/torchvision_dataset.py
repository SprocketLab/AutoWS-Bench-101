import json
from abc import ABC, abstractmethod
from pathlib import Path
from subprocess import check_output
from typing import List, Optional

import numpy as np
from torch import Generator
from torch.utils.data import Dataset, random_split
from torchvision import datasets

fwrench_datasets_root = Path(__file__).resolve().parent.parent.parent / "datasets"


class TorchVisionDataset(ABC):
    def __init__(
        self,
        dataset_name: str,
        input_root: Optional[Path] = None,
        output_root: Optional[Path] = None,
    ):
        self.dataset_name = dataset_name

        if input_root is None:
            input_root = fwrench_datasets_root / "torchvision_untransformed"
        if output_root is None:
            output_root = fwrench_datasets_root

        self.input_path = input_root / dataset_name
        self.output_path = output_root / dataset_name

        self.train_split: Dataset = None
        self.valid_split: Dataset = None
        self.test_split: Dataset = None

    def _append_output_path(self, suffix: str):
        self.output_path = self.output_path.with_name(self.output_path.name + suffix)

    @staticmethod
    def split(dataset: Dataset, train_p=0.95, seed=42):
        train_size = int(len(dataset) * train_p)
        valid_size = len(dataset) - train_size

        train_subset, valid_subset = random_split(
            dataset, [train_size, valid_size], generator=Generator().manual_seed(seed)
        )
        return train_subset, valid_subset

    def _transform(self):
        self._write_json("train", self.train_split)
        self._write_json("valid", self.valid_split)
        self._write_json("test", self.test_split)

    def _write_json(self, partition_name: str, data: Dataset):
        feature_output_path = self.output_path / partition_name

        self.output_path.mkdir(parents=True, exist_ok=True)
        feature_output_path.mkdir(parents=True, exist_ok=True)

        output_json = self.output_path / f"{partition_name}.json"

        with open(output_json, "w") as f:
            output = {}

            for i, (feature, label) in enumerate(data):
                key = str(i)

                feature = np.array(feature)

                if len(feature.shape) == 2:
                    feature = feature.reshape(1, *feature.shape)

                npy_path_feature = feature_output_path / f"{i}.npy"
                npy_filename_feature = npy_path_feature.name

                if not npy_path_feature.exists():
                    np.save(npy_path_feature, feature, allow_pickle=False)

                output[key] = {
                    "label": np.array(label).tolist(),
                    "data": {"feature": npy_filename_feature},
                    "weak_labels": [],
                }

            json.dump(output, f)

    def tar(self):
        npy_dirs = ["train", "valid", "test"]
        json_files = ["train.json", "valid.json", "test.json"]
        output_tar = self.output_path.with_suffix(".tar.gz")
        check_output(
            ["tar", "czf", output_tar, "-C", self.output_path] + npy_dirs + json_files
        )

    @abstractmethod
    def get_dataset(self, download=True):
        pass


class MNISTDataset(TorchVisionDataset):
    def __init__(
        self, input_root: Optional[Path] = None, output_root: Optional[Path] = None
    ):
        super().__init__("MNIST", input_root, output_root)

    def get_dataset(self, download=True, train_p=0.95, seed=42):
        trainvalid = datasets.MNIST(self.input_path, train=True, download=download)
        self.test_split = datasets.MNIST(
            self.input_path, train=False, download=download
        )
        self.train_split, self.valid_split = TorchVisionDataset.split(
            trainvalid, train_p=train_p, seed=seed
        )

        valid_size = len(self.valid_split)
        self._append_output_path(f"_{valid_size}")

        if download and not self.output_path.exists():
            self._transform()


class FashionMNISTDataset(TorchVisionDataset):
    def __init__(
        self, input_root: Optional[Path] = None, output_root: Optional[Path] = None
    ):
        super().__init__("FashionMNIST", input_root, output_root)

    def get_dataset(self, download=True, train_p=0.95, seed=42):
        trainvalid = datasets.FashionMNIST(
            self.input_path, train=True, download=download
        )
        self.test_split = datasets.FashionMNIST(
            self.input_path, train=False, download=download
        )
        self.train_split, self.valid_split = TorchVisionDataset.split(
            trainvalid, train_p=train_p, seed=seed
        )

        valid_size = len(self.valid_split)
        self._append_output_path(f"_{valid_size}")

        if download and not self.output_path.exists():
            self._transform()


class EMNISTDataset(TorchVisionDataset):
    def __init__(
        self, input_root: Optional[Path] = None, output_root: Optional[Path] = None
    ):
        super().__init__("EMNIST", input_root, output_root)

    def get_dataset(self, split="byclass", download=True, train_p=0.95, seed=42):
        trainvalid = datasets.EMNIST(
            self.input_path, split=split, train=True, download=download
        )
        self.test_split = datasets.EMNIST(
            self.input_path, split=split, train=False, download=download
        )
        self.train_split, self.valid_split = TorchVisionDataset.split(
            trainvalid, train_p=train_p, seed=seed
        )

        self._append_output_path(f"_{split}")

        valid_size = len(self.valid_split)
        self._append_output_path(f"_{valid_size}")

        if download and not self.output_path.exists():
            self._transform()


class KMNISTDataset(TorchVisionDataset):
    def __init__(
        self, input_root: Optional[Path] = None, output_root: Optional[Path] = None
    ):
        super().__init__("KMNIST", input_root, output_root)

    def get_dataset(self, download=True, train_p=0.95, seed=42):
        trainvalid = datasets.KMNIST(self.input_path, train=True, download=download)
        self.test_split = datasets.KMNIST(
            self.input_path, train=False, download=download
        )
        self.train_split, self.valid_split = TorchVisionDataset.split(
            trainvalid, train_p=train_p, seed=seed
        )

        valid_size = len(self.valid_split)
        self._append_output_path(f"_{valid_size}")

        if download and not self.output_path.exists():
            self._transform()


class QMNISTDataset(TorchVisionDataset):
    def __init__(
        self, input_root: Optional[Path] = None, output_root: Optional[Path] = None
    ):
        super().__init__("QMNIST", input_root, output_root)

    def get_dataset(self, download=True, train_p=0.95, seed=42):
        trainvalid = datasets.QMNIST(self.input_path, train=True, download=download)
        self.test_split = datasets.QMNIST(
            self.input_path, train=False, download=download
        )
        self.train_split, self.valid_split = TorchVisionDataset.split(
            trainvalid, train_p=train_p, seed=seed
        )

        valid_size = len(self.valid_split)
        self._append_output_path(f"_{valid_size}")

        if download and not self.output_path.exists():
            self._transform()


class SVHNDataset(TorchVisionDataset):
    def __init__(
        self, input_root: Optional[Path] = None, output_root: Optional[Path] = None
    ):
        super().__init__("SVHN", input_root, output_root)

    def get_dataset(self, download=True, train_p=0.95, seed=42):
        trainvalid = datasets.SVHN(self.input_path, split="train", download=download)
        self.test_split = datasets.SVHN(
            self.input_path, split="test", download=download
        )
        self.train_split, self.valid_split = TorchVisionDataset.split(
            trainvalid, train_p=train_p, seed=seed
        )

        valid_size = len(self.valid_split)
        self._append_output_path(f"_{valid_size}")

        if download and not self.output_path.exists():
            self._transform()


class CIFAR10Dataset(TorchVisionDataset):
    def __init__(
        self, input_root: Optional[Path] = None, output_root: Optional[Path] = None
    ):
        super().__init__("CIFAR10", input_root, output_root)

    def get_dataset(self, download=True, train_p=0.95, seed=42):
        trainvalid = datasets.CIFAR10(self.input_path, train=True, download=download)
        self.test_split = datasets.CIFAR10(
            self.input_path, train=False, download=download
        )
        self.train_split, self.valid_split = TorchVisionDataset.split(
            trainvalid, train_p=train_p, seed=seed
        )

        valid_size = len(self.valid_split)
        self._append_output_path(f"_{valid_size}")

        if download and not self.output_path.exists():
            self._transform()


class CIFAR100Dataset(TorchVisionDataset):
    def __init__(
        self, input_root: Optional[Path] = None, output_root: Optional[Path] = None
    ):
        super().__init__("CIFAR100", input_root, output_root)

    def get_dataset(self, download=True, train_p=0.95, seed=42):
        trainvalid = datasets.CIFAR100(self.input_path, train=True, download=download)
        self.test_split = datasets.CIFAR100(
            self.input_path, train=False, download=download
        )
        self.train_split, self.valid_split = TorchVisionDataset.split(
            trainvalid, train_p=train_p, seed=seed
        )

        valid_size = len(self.valid_split)
        self._append_output_path(f"_{valid_size}")

        if download and not self.output_path.exists():
            self._transform()

