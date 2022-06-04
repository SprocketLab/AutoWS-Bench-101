from abc import abstractmethod
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from shutil import copyfileobj
from typing import List, Optional, Union
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np

from .dataset import FWRENCHDataset


@dataclass
class Url:
    url: str
    md5: str


Split = str


class DownloadableDataset(FWRENCHDataset):
    @staticmethod
    def _check_md5(filepath: Path, chunk_size: int = 65536) -> str:
        with open(filepath, "rb") as f:
            checksum = md5()

            while True:
                chunk = f.read(chunk_size)
                if len(chunk) == 0:
                    break
                checksum.update(chunk)

            return checksum.hexdigest()

    def _url_to_download_filepath(self, url: Url):
        filename = Path(urlparse(url.url).path).name
        download_filepath = self.download_path / filename

        return download_filepath

    def __init__(
        self,
        split: Split,
        name: Optional[str] = None,
        path: Optional[Union[Path, str]] = None,
        download: bool = True,
        download_path: Optional[Union[Path, str]] = None,
    ):
        if name is None:
            name = type(self).__name__
        super().__init__(name, split, path, download, download_path)

    @property
    @abstractmethod
    def urls(self) -> List[Url]:
        pass

    def download(self):
        self.download_path.mkdir(parents=True, exist_ok=True)

        for url in self.urls:
            filename = Path(urlparse(url.url).path).name
            download_filepath = self.download_path / filename

            if not download_filepath.exists():
                download_file = open(download_filepath, "wb")
                download_request = urlopen(url.url)
                copyfileobj(download_request, download_file)

            checksum = DownloadableDataset._check_md5(download_filepath)
            if checksum != url.md5:
                raise RuntimeError(
                    f"checksum of {filename} does not match\n"
                    f"{checksum} != {url.md5}"
                )


class NinaProDB5(DownloadableDataset):

    _urls = {
        "train": {
            "feature": Url(
                "https://pde-xd.s3.amazonaws.com/ninapro/ninapro_train.npy",
                "d4c33785587983348e6091e86a0d30b6",
            ),
            "label": Url(
                "https://pde-xd.s3.amazonaws.com/ninapro/label_train.npy",
                "4164e7fcae3f8abed0a04fd86b704f77",
            ),
        },
        "valid": {
            "feature": Url(
                "https://pde-xd.s3.amazonaws.com/ninapro/ninapro_val.npy",
                "138b106d4374a3fc204e41f6c4deda49",
            ),
            "label": Url(
                "https://pde-xd.s3.amazonaws.com/ninapro/label_val.npy",
                "f0464819f7ed56cb1cbbfafd312bd604",
            ),
        },
        "test": {
            "feature": Url(
                "https://pde-xd.s3.amazonaws.com/ninapro/ninapro_test.npy",
                "8a2b2cf87d5a5cf3986db2d67ebcfa18",
            ),
            "label": Url(
                "https://pde-xd.s3.amazonaws.com/ninapro/label_test.npy",
                "807b502000dba69ae0111063970ac5d2",
            ),
        },
    }

    @property
    def urls(self):
        return type(self)._urls[self.split].values()

    def transform(self):
        feature_download_filepath = self._url_to_download_filepath(
            type(self)._urls[self.split]["feature"]
        )
        label_download_filepath = self._url_to_download_filepath(
            type(self)._urls[self.split]["label"]
        )

        feature = np.load(feature_download_filepath)
        label = np.load(label_download_filepath)

        self.data = FWRENCHDataset._FeatureLabel(feature, label)

        self.write_meta()
        self.write_split()
