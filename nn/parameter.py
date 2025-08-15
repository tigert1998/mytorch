from tensor import Tensor
import torch.nn as nn


class Parameter(Tensor):
    def __init__(self, tensor: Tensor):
        super().__init__(tensor=tensor)
