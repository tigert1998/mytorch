import numpy as np

from mytorch.nn.module import Module
from mytorch.nn.parameter import Parameter, Tensor


class BatchNorm2d(Module):
    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device="cpu",
        dtype=np.float32,
    ):
        pass

    def forward(self, x): ...
