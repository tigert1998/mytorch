import struct
import os.path as osp
import requests
import gzip
import os

import numpy as np
from tqdm import tqdm
from PIL import Image

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


class MNIST(Dataset):
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
        self.train = train
        os.makedirs(osp.join(self.root, "MNIST/raw"), exist_ok=True)
        if download:
            self._download_and_save_npy()
        self._load_npy()
        self._transform = transform
        self._target_transform = target_transform

    def _load_npy(self):
        prefix = "train" if self.train else "test"
        self.x = np.load(osp.join(self.root, "MNIST", f"{prefix}_x.npy"))
        self.y = np.load(osp.join(self.root, "MNIST", f"{prefix}_y.npy"))

    def _download_and_save_npy(self):
        folder = osp.join(self.root, "MNIST/raw")
        for filename in self.FILENAMES.values():
            _download_single_file(filename, folder)
        for train in [False, True]:
            prefix = "train" if train else "test"
            imgs = []
            labels = []
            x_buf, y_buf = self._read_bytes(folder=folder, train=train)
            size = struct.unpack(">i", x_buf[4:8])[0]
            for i in tqdm(range(size)):
                imgs.append(self._get_img(x_buf, i).reshape((1, 28, 28)))
                labels.append(self._get_label(y_buf, i))
            np.save(
                osp.join(self.root, "MNIST", f"{prefix}_x.npy"),
                np.concatenate(imgs, axis=0),
            )
            np.save(osp.join(self.root, "MNIST", f"{prefix}_y.npy"), np.array(labels))

    @classmethod
    def _read_bytes(cls, folder, train):
        prefix = "train" if train else "test"
        with open(osp.join(folder, cls.FILENAMES[f"{prefix}_x"][:-3]), "rb") as f:
            x_buf = f.read()
        with open(osp.join(folder, cls.FILENAMES[f"{prefix}_y"][:-3]), "rb") as f:
            y_buf = f.read()
        return x_buf, y_buf

    @staticmethod
    def _get_img(x_buf, idx):
        offset = idx * 784
        data = struct.unpack_from(">784B", x_buf, offset + 16)
        return np.array(data).reshape(28, 28)

    @staticmethod
    def _get_label(y_buf, idx):
        label = struct.unpack_from(">B", y_buf, idx + 8)
        return label[0]

    def __getitem__(self, index):
        img = self.x[index]
        img = Image.fromarray(img.astype(np.uint8))
        if self._transform is not None:
            img = self._transform(img)

        label = self.y[index]
        if self._target_transform is not None:
            label = self._target_transform(label)
        return img, label

    def __len__(self):
        return self.x.shape[0]
