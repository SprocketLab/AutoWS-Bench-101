from abc import ABC, abstractmethod
from dataclasses import dataclass
from json import dump, load
from pathlib import Path
from typing import Optional, Union

import numpy as np
from torch.utils.data import Dataset as TorchDataset


class FWRENCHDataset(ABC, TorchDataset):

    DATASET_PATH = Path(__file__).resolve(strict=True).parents[2] / "datasets"

    META_JSON_FILENAME = "meta.json"
    LABEL_JSON_FILENAME = "label.json"

    SEED = 42

    @staticmethod
    def _split_json_filename(split):
        return f"{split}.json"

    @staticmethod
    def generate_split_indices(N, train_p, seed=SEED):
        train_size = int(N * train_p)
        indices = np.arange(N)
        train_indices = np.random.default_rng(seed).choice(
            indices, size=train_size, replace=False
        )
        valid_indices = np.setdiff1d(indices, train_indices)
        return train_indices, valid_indices

    @dataclass
    class _FeatureLabel:
        feature: np.ndarray
        label: np.ndarray

        def __getitem__(self, index: int):
            return self.feature[index], self.label[index]

        def __len__(self):
            assert len(self.feature) == len(self.label)
            return len(self.feature)

    def __init__(
        self,
        name: str,
        split: str,
        path: Optional[Union[Path, str]] = None,
        download: bool = True,
        download_path: Optional[Union[Path, str]] = None,
    ):
        super().__init__()

        self.split = split

        if path is None:
            self.path = FWRENCHDataset.DATASET_PATH / name
        else:
            self.path = Path(path) / name

        if download_path is None:
            self.download_path = FWRENCHDataset.DATASET_PATH / name
        else:
            self.download_path = Path(download_path) / name

        if download:
            self.download()

        if self._should_transform():
            self.transform()
        else:
            self.load()

    def _should_transform(self):
        return not (
            (self.path / FWRENCHDataset._split_json_filename(self.split)).exists()
            and (self.path / self.split).exists()
            and (self.path / FWRENCHDataset.META_JSON_FILENAME).exists()
        )

    def load(self):
        meta_json_filepath = self.path / FWRENCHDataset.META_JSON_FILENAME
        split_json_filepath = self.path / FWRENCHDataset._split_json_filename(
            self.split
        )
        split_feature_dir = self.path / self.split

        with open(meta_json_filepath) as meta_json_file:
            meta_json = load(meta_json_file)
            feature_dtype = getattr(np, meta_json["feature"]["dtype"])
            feature_shape = meta_json["feature"]["shape"]
            label_dtype = getattr(np, meta_json["label"]["dtype"])
            label_shape = meta_json["label"]["shape"]

            split_size = meta_json[self.split]["size"]

        with open(split_json_filepath) as split_json_file:
            split_json = load(split_json_file)
            feature = np.zeros((split_size, *feature_shape), dtype=feature_dtype)
            label = np.zeros((split_size, *label_shape), dtype=label_dtype)
            weak_labels = [None] * split_size
            for k, v in split_json.items():
                k = int(k)
                feature[k] = np.load(split_feature_dir / v["data"]["feature"])
                label[k] = v["label"]
                weak_labels[k] = np.array(v["weak_labels"])
        self.data = FWRENCHDataset._FeatureLabel(feature, label)
        self.weak_labels = np.array(weak_labels)
        


    def __getitem__(self, index: int):
        if isinstance(self.data, FWRENCHDataset._FeatureLabel):
            return self.data[index]
        return tuple(np.array(d) for d in self.data[index])

    @abstractmethod
    def download(self):
        pass

    @abstractmethod
    def transform(self):
        pass

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
