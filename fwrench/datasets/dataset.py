from dataclasses import dataclass
from json import load
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset as TorchDataset


class FWRENCHDataset(TorchDataset):

    DATASET_PATH = Path(__file__).resolve(strict=True).parents[2] / "datasets"

    META_JSON_FILENAME = "meta.json"
    LABEL_JSON_FILENAME = "label.json"

    SEED = 42

    @staticmethod
    def _split_json_filename(split):
        return f"{split}.json"

    @dataclass
    class __FeatureLabel:
        feature: NDArray
        label: NDArray

        def __getitem__(self, index: int):
            return self.feature[index], self.label[index]

        def __len__(self):
            assert len(self.feature) == len(self.label)
            return len(self.feature)

    def __init__(
        self,
        name: str,
        split: str,
        path: Optional[Path | str] = None,
        download: bool = True,
        download_path: Optional[Path | str] = None,
    ):
        super().__init__()

        self.split = split

        if path is None:
            self.path = FWRENCHDataset.DATASET_PATH / name
        else:
            self.path = Path(path) / name

        if download_path is None:
            download_path = FWRENCHDataset.DATASET_PATH / name
        else:
            download_path = Path(download_path) / name

        if download:
            self.download(download_path)

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

        self.data = FWRENCHDataset.__FeatureLabel(feature, label)
        self.weak_labels = np.array(weak_labels)

    def __getitem__(self, index: int):
        if isinstance(self.data, FWRENCHDataset.__FeatureLabel):
            return self.data[index]
        return tuple(np.array(d) for d in self.data[index])

    def download(self, download_path: Path):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError
