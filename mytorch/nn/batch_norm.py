import numpy as np

from mytorch.tensor import Tensor
from mytorch.nn.module import Module
from mytorch.nn.parameter import Parameter, Tensor
from mytorch.dtype import float32


class _BatchNormBase(Module):
    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device="cpu",
        dtype=float32,
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = Parameter(
                Tensor(shape=(num_features,), dtype=dtype, device=device)
            )
            self.bias = Parameter(
                Tensor(shape=(num_features,), dtype=dtype, device=device)
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.running_mean = Tensor(
                shape=(num_features,), dtype=dtype, device=device
            )
            self.running_var = Tensor(shape=(num_features,), dtype=dtype, device=device)
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.track_running_stats:
            self.running_mean.fill_(0)
            self.running_var.fill_(1)
        if self.affine:
            self.weight.fill_(1)
            self.bias.fill_(0)

    def forward(self, x: Tensor):
        reduce_axis = (0, *range(2, len(x.shape)))
        reshape_shape = (1, self.num_features, *([1] * len(x.shape[2:])))
        if self.training:
            batch_mean = x.mean(dim=reduce_axis, keepdim=True)
            batch_var = x.var(dim=reduce_axis, keepdim=True, correction=0)

            if self.track_running_stats:
                self.running_mean.copy_(
                    (1 - self.momentum) * self.running_mean
                    + self.momentum * batch_mean.reshape((self.num_features,))
                )
                self.running_var.copy_(
                    (1 - self.momentum) * self.running_var
                    + self.momentum * batch_var.reshape((self.num_features,))
                )

            x_normalized = (x - batch_mean) / (batch_var + self.eps) ** 0.5
        else:
            if self.track_running_stats:
                mean = self.running_mean.reshape(reshape_shape)
                var = self.running_var.reshape(reshape_shape)
            else:
                mean = x.mean(dim=reduce_axis, keepdim=True)
                var = x.var(dim=reduce_axis, keepdim=True, correction=0)

            x_normalized = (x - mean) / (var + self.eps) ** 0.5

        if self.affine:
            weight = self.weight.reshape(reshape_shape)
            bias = self.bias.reshape(reshape_shape)
            return weight * x_normalized + bias
        else:
            return x_normalized


class BatchNorm1d(_BatchNormBase):
    def __init__(
        self,
        num_features,
        eps=0.00001,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device="cpu",
        dtype=float32,
    ):
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype
        )


class BatchNorm2d(_BatchNormBase):
    def __init__(
        self,
        num_features,
        eps=0.00001,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device="cpu",
        dtype=float32,
    ):
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype
        )

    def forward(self, x):
        from mytorch.nn.functional.batch_norm import _batch_norm_2d

        return _batch_norm_2d(
            x,
            self.weight,
            self.bias,
            self.eps,
            self.training,
            self.momentum,
            self.running_mean,
            self.running_var,
        )
