from typing import Optional

import numpy as np
import numpy.typing as npt

from mytorch.tensor import Tensor, shape_size
from mytorch.backends.backend_dispatcher import BackendDispatcher


@BackendDispatcher.instance().register_backend_function("cpu", "batch_norm2d")
def cpu_batch_norm2d(
    x: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    eps: float,
    training: bool,
    momentum: float,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
):
    num_features = x.shape[1]
    track_running_stats = running_mean is not None and running_var is not None
    affine = weight is not None and bias is not None

    reduce_axis = (0, *range(2, len(x.shape)))
    reshape_shape = (1, num_features, *([1] * len(x.shape[2:])))

    x_np: npt.NDArray = x._numpy()
    if training:
        mean = x_np.mean(axis=reduce_axis, keepdims=True)
        var = x_np.var(axis=reduce_axis, keepdims=True, ddof=0)
        if track_running_stats:
            running_mean.copy_(
                (1 - momentum) * running_mean + momentum * mean.reshape((num_features,))
            )
            running_var.copy_(
                (1 - momentum) * running_var + momentum * var.reshape((num_features,))
            )
        x_normalized = (x_np - mean) / np.sqrt(var + eps)
    else:
        if track_running_stats:
            mean = running_mean._numpy().reshape(reshape_shape)
            var = running_var._numpy().reshape(reshape_shape)
        else:
            mean = x_np.mean(axis=reduce_axis, keepdims=True)
            var = x_np.var(axis=reduce_axis, keepdims=True, ddof=0)
        x_normalized = (x_np - mean) / np.sqrt(var + eps)
    if affine:
        weight = weight._numpy().reshape(reshape_shape)
        bias = bias._numpy().reshape(reshape_shape)
        return Tensor(weight * x_normalized + bias), Tensor(mean), Tensor(var)
    else:
        return Tensor(x_normalized), Tensor(mean), Tensor(var)


@BackendDispatcher.instance().register_backend_function("cpu", "batch_norm2d_backward")
def cpu_batch_norm2d_backward(
    output_grad: Tensor,
    mean: Tensor,
    var: Tensor,
    x: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    eps: float,
):
    num_features = x.shape[1]
    reduce_axis = (0, *range(2, len(x.shape)))
    loop_size = shape_size([x.shape[i] for i in reduce_axis])
    reshape_shape = (1, num_features, *([1] * len(x.shape[2:])))

    affine = weight is not None and bias is not None

    x_np = x._numpy()
    mean_np = mean._numpy()
    var_np = var._numpy()
    output_grad_np = output_grad._numpy()
    x_normalized = (x_np - mean_np) / np.sqrt(var_np + eps)

    if affine:
        bias_grad = Tensor(output_grad_np.sum(reduce_axis, dtype=x.dtype.np_dtype))
        weight_grad = Tensor(
            (output_grad_np * x_normalized).sum(reduce_axis, dtype=x.dtype.np_dtype)
        )
        x_normalized_grad = output_grad_np * weight._numpy().reshape(reshape_shape)
    else:
        bias_grad = weight_grad = None
        x_normalized_grad = output_grad_np

    var_grad = (
        x_normalized_grad * -0.5 * (x_np - mean_np) * (var_np + eps) ** -1.5
    ).sum(axis=reduce_axis, keepdims=True)
    mean_grad = (x_normalized_grad * -((var_np + eps) ** -0.5)).sum(
        axis=reduce_axis, keepdims=True
    )
    x_grad = x_normalized_grad * (var_np + eps) ** -0.5
    x_grad += mean_grad / loop_size
    x_grad += var_grad * 2 / loop_size * (x_np - mean_np)
    return Tensor(x_grad), weight_grad, bias_grad
