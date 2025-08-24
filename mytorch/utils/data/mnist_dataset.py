import struct
import os.path as osp
import requests
import gzip
import os

import numpy as np

from mytorch.utils.data.dataset import Dataset


def _download_single_file(filename: str, folder: str):
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    file = requests.get(url=base_url + filename)
    path = osp.join(folder, filename)
    with open(path, "wb+") as f:
        f.write(file.content)
    cur_path = path.replace(".gz", "")
    g_file = gzip.GzipFile(path)
    with open(cur_path, "wb+") as f:
        f.write(g_file.read())


class MNISTDataset(Dataset):
    FILENAMES = {
        "train_x": "train-images-idx3-ubyte.gz",
        "train_y": "train-labels-idx1-ubyte.gz",
        "test_x": "t10k-images-idx3-ubyte.gz",
        "test_y": "t10k-labels-idx1-ubyte.gz",
    }

    def __init__(
        self, root, train=True, download=False, transform=None, target_transform=None
    ):
        super().__init__()
        self.root = root
        os.makedirs(osp.join(self.root, "MNIST"), exist_ok=True)
        if download:
            self._download()
        self._read_bytes(train)
        self._size = struct.unpack(">i", self.x_buf[4:8])[0]
        self._transform = transform
        self._target_transform = target_transform

    def _read_bytes(self, train):
        prefix = "train" if train else "test"
        with open(
            osp.join(self.root, "MNIST", self.FILENAMES[f"{prefix}_x"][:-3]), "rb"
        ) as f:
            self.x_buf = f.read()
        with open(
            osp.join(self.root, "MNIST", self.FILENAMES[f"{prefix}_y"][:-3]), "rb"
        ) as f:
            self.y_buf = f.read()

    def _download(self):
        for filename in self.FILENAMES.values():
            _download_single_file(filename, osp.join(self.root, "MNIST"))

    def _get_img(self, idx):
        offset = idx * 784
        data = struct.unpack_from(">784B", self.x_buf[16:], offset)
        return np.array(data).reshape(28, 28)

    def _get_label(self, idx):
        label = struct.unpack_from(">B", self.y_buf, idx + 8)
        return label[0]

    def __getitem__(self, index):
        img = self._get_img(index)
        if self._transform is not None:
            img = self._transform(img)

        label = self._get_label(index)
        if self._target_transform is not None:
            label = self._target_transform(label)
        return img, label

    def __len__(self):
        return self._size
