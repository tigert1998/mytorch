from PIL import Image
import numpy as np

from mytorch.nn.module import Module
from mytorch.tensor import Tensor


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Normalize(Module):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std


class ToTensor:
    def __call__(self, pic: Image.Image):
        if pic.mode not in ["L"]:
            raise NotImplementedError()
        data = (np.array(pic) / 255.0).astype(np.float32)
        if len(data.shape) == 2:
            data = np.expand_dims(data, 0)
        elif len(data.shape) == 3:
            data = np.transpose(data, (2, 0, 1))
        return Tensor(cpu_array=data)
