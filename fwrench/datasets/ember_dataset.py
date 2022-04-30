import os
import tarfile
from json import loads
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import numpy as np
from ember import PEFeatureExtractor
from tqdm import tqdm

from dataset import FWRENCHDataset
from downloadable_dataset import DownloadableDataset, Split, Url

Year = str


class EmberDataset(DownloadableDataset):

    # Elements from https://github.com/elastic/ember/blob/master/ember/__init__.py

    _urls = {
        "2017": Url(
            "https://ember.elastic.co/ember_dataset_2017_2.tar.bz2",
            "255e23a8d8afad967a30b2e188bd4213",
        ),
        "2018": Url(
            "https://ember.elastic.co/ember_dataset_2018_2.tar.bz2",
            "4548351419e305c979497e4a40463b7f",
        ),
    }

    _extract_path = {"2017": "ember_2017_2", "2018": "ember2018"}

    @staticmethod
    def raw_feature_iterator(json_filepaths):
        for path in json_filepaths:
            with open(path, "r") as f:
                for line in f:
                    yield loads(line)

    @staticmethod
    def rows(filepath):
        with open(filepath) as f:
            return sum(1 for _ in f)

    @staticmethod
    def feature_extractor(raw_feature_path):
        extractor = PEFeatureExtractor()
        n_samples = EmberDataset.rows(raw_feature_path)
        X = np.zeros(shape=(n_samples, extractor.dim), dtype=np.float32)
        y = np.zeros(shape=n_samples, dtype=np.float32)

        with open(raw_feature_path) as raw_feature:
            for i, raw_feature_entry in enumerate(raw_feature):
                X[i] = extractor.process_raw_features(raw_feature_entry)
                y[i] = raw_feature_entry["label"]

        return X, y

    @staticmethod
    def vectorize(X_path, y_path, raw_feature_paths):
        extractor = PEFeatureExtractor()
        n_samples = sum(map(EmberDataset.rows, raw_feature_paths))

        if X_path.exists() and y_path.exists():
            X = np.memmap(
                X_path, dtype=np.float32, mode="r", shape=(n_samples, extractor.dim)
            )
            y = np.memmap(y_path, dtype=np.float32, mode="r", shape=n_samples)
        else:
            X = np.memmap(
                X_path, dtype=np.float32, mode="w+", shape=(n_samples, extractor.dim)
            )
            y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=n_samples)

            for i, raw_features in tqdm(
                enumerate(EmberDataset.raw_feature_iterator(raw_feature_paths)),
                total=n_samples,
            ):
                X[i] = extractor.process_raw_features(raw_features)
                y[i] = raw_features["label"]

        return X, y

    def __init__(
        self,
        year: Year,
        split: Split,
        name: Optional[str] = None,
        path: Optional[Union[Path, str]] = None,
        download: bool = True,
        download_path: Optional[Union[Path, str]] = None,
        train_p: float = 0.95,
    ):
        assert year in EmberDataset._urls
        self.year = year
        self.train_p = train_p
        if name is None:
            name = f"Ember{year}"
        super().__init__(split, name, path, download, download_path)

    @property
    def urls(self):
        return [type(self)._urls[self.year]]

    def transform(self):
        bzip_tarfile_path = (
            self.path / Path(urlparse(EmberDataset._urls[self.year].url).path).name
        )
        extract_path = self.path / EmberDataset._extract_path[self.year]

        if not extract_path.exists():
            with tarfile.open(bzip_tarfile_path, "r:bz2") as bzip_tarfile:
                bzip_tarfile.extractall(self.path)

        if self.split in ("train", "valid"):
            X_path = extract_path / "X_trainvalid.dat"
            y_path = extract_path / "y_trainvalid.dat"

            raw_feature_paths = [
                extract_path / f"train_features_{i}.jsonl" for i in range(6)
            ]

            EmberDataset.vectorize(X_path, y_path, raw_feature_paths)

            X, y = EmberDataset.vectorize(X_path, y_path, raw_feature_paths)

            train_indices, valid_indices = FWRENCHDataset.generate_split_indices(
                len(X), self.train_p
            )
            indices = train_indices if self.split == "train" else valid_indices

            X = X[indices]
            y = y[indices]

            self.data = FWRENCHDataset._FeatureLabel(X, y)

        elif self.split == "test":
            X_path = extract_path / "X_test.dat"
            y_path = extract_path / "y_test.dat"

            raw_feature_paths = [extract_path / "test_features.jsonl"]

            X, y = EmberDataset.vectorize(X_path, y_path, raw_feature_paths)
            self.data = FWRENCHDataset._FeatureLabel(X, y)
        else:
            raise ValueError(f"Invalid split: {self.split}")

        self.write_meta()
        self.write_split()
