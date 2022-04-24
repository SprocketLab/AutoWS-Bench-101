from json import dump, load
from pathlib import Path
from typing import Optional

import numpy as np
from torch import Generator
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import random_split
from torchvision import datasets

from dataset import FWRENCHDataset


class TorchVisionDataset(FWRENCHDataset):

    DEFAULT_TORCHVISION_DOWNLOAD_PATH = (
        FWRENCHDataset.DATASET_PATH / "torchvision_untransformed"
    )

    def __init__(
        self,
        name: str,
        split: str,
        path: Optional[Path | str] = None,
        download: bool = True,
        download_path: Optional[Path | str] = None,
    ):
        if download_path is None:
            download_path = TorchVisionDataset.DEFAULT_TORCHVISION_DOWNLOAD_PATH

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

    def write_meta(self):
        self.path.mkdir(parents=True, exist_ok=True)

        meta_json_path = self.path / FWRENCHDataset.META_JSON_FILENAME

        if meta_json_path.exists():
            with open(meta_json_path) as meta_json_file:
                meta_json = load(meta_json_file)
        else:
            meta_json = dict()

        meta_json[self.split] = dict()
        meta_json[self.split]["size"] = len(self.data)
        if "feature" not in meta_json:
            meta_json["feature"] = dict()
            meta_json["feature"]["shape"] = list(self[0][0].shape)
            meta_json["feature"]["dtype"] = str(np.dtype(self[0][0].dtype))
        if "label" not in meta_json:
            meta_json["label"] = dict()
            meta_json["label"]["shape"] = list(self[0][1].shape)
            meta_json["label"]["dtype"] = str(np.dtype(self[0][1].dtype))

        with open(meta_json_path, "w") as meta_json_file:
            dump(meta_json, meta_json_file)

        label_json_path = self.path / FWRENCHDataset.LABEL_JSON_FILENAME

        if not label_json_path.exists():
            unique_labels = np.unique(
                [self[i][1] for i in range(len(self.data))]
            ).tolist()
            label_json = dict(zip(map(str, unique_labels), unique_labels))

            with open(label_json_path, "w") as label_json_file:
                dump(label_json, label_json_file)

    def write_split(self):
        self.path.mkdir(parents=True, exist_ok=True)

        split_feature_dir = self.path / self.split
        split_feature_dir.mkdir(parents=True, exist_ok=True)

        output_json_path = self.path / FWRENCHDataset._split_json_filename(self.split)

        output = {}

        for i, (feature, label) in enumerate(self.data):
            key = str(i)

            feature = np.array(feature)

            if len(feature.shape) == 2:
                feature = feature.reshape(1, *feature.shape)

            npy_path_feature = split_feature_dir / f"{i}.npy"
            npy_filename_feature = npy_path_feature.name

            if not npy_path_feature.exists():
                np.save(npy_path_feature, feature, allow_pickle=False)

            output[key] = {
                "label": np.array(label).tolist(),
                "data": {"feature": npy_filename_feature},
                "weak_labels": [],
            }

        with open(output_json_path, "w") as f:
            dump(output, f)


class MNISTDataset(TorchVisionDataset):
    TRAIN_P = 0.95

    def __init__(
        self,
        split: str,
        name: str = "MNIST",
        path: Optional[Path | str] = None,
        download: bool = True,
        download_path: Optional[Path | str] = None,
    ):
        super().__init__(name, split, path, download, download_path)

    def download(self, download_path: Path):
        trainvalid = datasets.MNIST(download_path, train=True, download=True)
        train_split, valid_split = TorchVisionDataset._split(
            trainvalid, MNISTDataset.TRAIN_P
        )
        test_split = datasets.MNIST(download_path, train=False, download=True)

        valid_size = len(valid_split)
        self._set_path_suffix(valid_size)
        self._set_data(train_split, valid_split, test_split)


class FashionMNISTDataset(TorchVisionDataset):
    TRAIN_P = 0.95

    def __init__(
        self,
        split: str,
        name: str = "FashionMNIST",
        path: Optional[Path | str] = None,
        download: bool = True,
        download_path: Optional[Path | str] = None,
    ):
        super().__init__(name, split, path, download, download_path)

    def download(self, download_path: Path):
        trainvalid = datasets.FashionMNIST(download_path, train=True, download=True)
        train_split, valid_split = TorchVisionDataset._split(
            trainvalid, train_p=FashionMNISTDataset.TRAIN_P
        )
        test_split = datasets.FashionMNIST(download_path, train=False, download=True)

        valid_size = len(valid_split)
        self._set_path_suffix(valid_size)
        self._set_data(train_split, valid_split, test_split)


class EMNISTDataset(TorchVisionDataset):
    TRAIN_P = 0.95

    def __init__(
        self,
        emnist_split: str,
        split: str,
        name: str = "EMNIST",
        path: Optional[Path | str] = None,
        download: bool = True,
        download_path: Optional[Path | str] = None,
    ):
        self.emnist_split = emnist_split
        super().__init__(name, split, path, download, download_path)

    def download(self, download_path: Path):
        self._set_path_suffix(self.emnist_split)

        trainvalid = datasets.EMNIST(
            download_path, split=self.emnist_split, train=True, download=True
        )
        train_split, valid_split = TorchVisionDataset._split(
            trainvalid, train_p=EMNISTDataset.TRAIN_P
        )
        test_split = datasets.EMNIST(
            download_path, split=self.emnist_split, train=False, download=True
        )

        valid_size = len(valid_split)
        self._set_path_suffix(valid_size)
        self._set_data(train_split, valid_split, test_split)


class KMNISTDataset(TorchVisionDataset):
    TRAIN_P = 0.95

    def __init__(
        self,
        split: str,
        name: str = "KMNIST",
        path: Optional[Path | str] = None,
        download: bool = True,
        download_path: Optional[Path | str] = None,
    ):
        super().__init__(name, split, path, download, download_path)

    def download(self, download_path: Path):
        trainvalid = datasets.KMNIST(download_path, train=True, download=True)
        train_split, valid_split = TorchVisionDataset._split(
            trainvalid, train_p=KMNISTDataset.TRAIN_P
        )
        test_split = datasets.KMNIST(download_path, train=False, download=True)

        valid_size = len(valid_split)
        self._set_path_suffix(valid_size)
        self._set_data(train_split, valid_split, test_split)


class QMNISTDataset(TorchVisionDataset):
    TRAIN_P = 0.95

    def __init__(
        self,
        split: str,
        name: str = "QMNIST",
        path: Optional[Path | str] = None,
        download: bool = True,
        download_path: Optional[Path | str] = None,
    ):
        super().__init__(name, split, path, download, download_path)

    def download(self, download_path: Path):
        trainvalid = datasets.QMNIST(download_path, train=True, download=True)
        train_split, valid_split = TorchVisionDataset._split(
            trainvalid, train_p=QMNISTDataset.TRAIN_P
        )
        test_split = datasets.QMNIST(download_path, train=False, download=True)

        valid_size = len(valid_split)
        self._set_path_suffix(valid_size)
        self._set_data(train_split, valid_split, test_split)


class SVHNDataset(TorchVisionDataset):
    TRAIN_P = 0.95

    def __init__(
        self,
        split: str,
        name: str = "SVHN",
        path: Optional[Path | str] = None,
        download: bool = True,
        download_path: Optional[Path | str] = None,
    ):
        super().__init__(name, split, path, download, download_path)

    def download(self, download_path: Path):
        trainvalid = datasets.SVHN(download_path, split="train", download=True)
        train_split, valid_split = TorchVisionDataset._split(
            trainvalid, train_p=SVHNDataset.TRAIN_P
        )
        test_split = datasets.SVHN(download_path, split="test", download=True)

        valid_size = len(valid_split)
        self._set_path_suffix(valid_size)
        self._set_data(train_split, valid_split, test_split)


class CIFAR10Dataset(TorchVisionDataset):
    TRAIN_P = 0.95

    def __init__(
        self,
        split: str,
        name: str = "CIFAR10",
        path: Optional[Path | str] = None,
        download: bool = True,
        download_path: Optional[Path | str] = None,
    ):
        super().__init__(name, split, path, download, download_path)

    def download(self, download_path: Path):
        trainvalid = datasets.CIFAR10(download_path, train=True, download=True)
        train_split, valid_split = TorchVisionDataset._split(
            trainvalid, train_p=CIFAR10Dataset.TRAIN_P
        )
        test_split = datasets.CIFAR10(download_path, train=False, download=True)

        valid_size = len(valid_split)
        self._set_path_suffix(valid_size)
        self._set_data(train_split, valid_split, test_split)


class CIFAR100Dataset(TorchVisionDataset):
    TRAIN_P = 0.95

    def __init__(
        self,
        split: str,
        name: str = "CIFAR100",
        path: Optional[Path | str] = None,
        download: bool = True,
        download_path: Optional[Path | str] = None,
    ):
        super().__init__(name, split, path, download, download_path)

    def download(self, download_path: Path):
        trainvalid = datasets.CIFAR100(download_path, train=True, download=True)
        train_split, valid_split = TorchVisionDataset._split(
            trainvalid, train_p=CIFAR100Dataset.TRAIN_P
        )
        test_split = datasets.CIFAR100(download_path, train=False, download=True)

        valid_size = len(valid_split)
        self._set_path_suffix(valid_size)
        self._set_data(train_split, valid_split, test_split)


def main():
    MNISTDataset("train")
    MNISTDataset("valid")
    MNISTDataset("test")

    FashionMNISTDataset("train")
    FashionMNISTDataset("valid")
    FashionMNISTDataset("test")

    EMNISTDataset("byclass", "train")
    EMNISTDataset("byclass", "valid")
    EMNISTDataset("byclass", "test")

    KMNISTDataset("train")
    KMNISTDataset("valid")
    KMNISTDataset("test")

    QMNISTDataset("train")
    QMNISTDataset("valid")
    QMNISTDataset("test")

    SVHNDataset("train")
    SVHNDataset("valid")
    SVHNDataset("test")

    CIFAR10Dataset("train")
    CIFAR10Dataset("valid")
    CIFAR10Dataset("test")

    CIFAR100Dataset("train")
    CIFAR100Dataset("valid")
    CIFAR100Dataset("test")


if __name__ == "__main__":
    main()
