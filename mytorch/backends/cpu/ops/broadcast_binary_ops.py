import numpy as np
import numpy.typing as npt
from typing import Tuple, List

from mytorch.backends.utils import calculate_broadcast_shape
from mytorch.backends.backend_dispatcher import BackendDispatcher
from mytorch.tensor import Tensor


def _tile_tensor_cpu(x, output_grad_shape) -> Tuple[npt.NDArray, List[int]]:
    x_shape = (1,) * (len(output_grad_shape) - len(x.shape)) + x.shape
    x_axis = [i for i in range(len(x_shape)) if x_shape[i] < output_grad_shape[i]]
    x_tile_reps = [
        output_grad_shape[i] if x_shape[i] < output_grad_shape[i] else 1
        for i in range(len(x_shape))
    ]
    x_tile = np.tile(x, x_tile_reps)
    return x_tile, x_axis


def _backward_cpu(output_grad: Tensor, x: Tensor, y: Tensor, func):
    x_grad = Tensor(dtype=x.dtype, shape=x.shape, device=x.device)
    x_grad.fill_(0)
    y_grad = Tensor(dtype=y.dtype, shape=y.shape, device=y.device)
    y_grad.fill_(0)

    x_tile, x_axis = _tile_tensor_cpu(x._numpy(), output_grad.shape)
    y_tile, y_axis = _tile_tensor_cpu(y._numpy(), output_grad.shape)
    x_grad_cpu_array, y_grad_cpu_array = func(x_tile, y_tile, output_grad._numpy())
    x_grad._cpu_array = x_grad_cpu_array.sum(axis=tuple(x_axis)).reshape(x.shape)
    y_grad._cpu_array = y_grad_cpu_array.sum(axis=tuple(y_axis)).reshape(y.shape)

    return x_grad, y_grad


@BackendDispatcher.instance().register_backend_function("cpu", "add")
def cpu_add(x: Tensor, y: Tensor, alpha):
    shape = calculate_broadcast_shape(x.shape, y.shape)
    output_tensor = Tensor(
        dtype=x.dtype,
        shape=shape,
        device=x.device,
    )
    output_tensor._cpu_array = x._numpy() + y._numpy() * alpha
    return output_tensor


@BackendDispatcher.instance().register_backend_function("cpu", "add_backward")
def cpu_add_backward(output_grad: Tensor, x: Tensor, y: Tensor, alpha):
    def func(x, y, output_grad):
        return output_grad, output_grad * alpha

    return _backward_cpu(output_grad, x, y, func)


@BackendDispatcher.instance().register_backend_function("cpu", "sub")
def cpu_sub(x: Tensor, y: Tensor, alpha):
    shape = calculate_broadcast_shape(x.shape, y.shape)
    output_tensor = Tensor(
        dtype=x.dtype,
        shape=shape,
        device=x.device,
    )
    output_tensor._cpu_array = x._numpy() - y._numpy() * alpha
    return output_tensor


@BackendDispatcher.instance().register_backend_function("cpu", "sub_backward")
def cpu_sub_backward(output_grad: Tensor, x: Tensor, y: Tensor, alpha):
    def func(x, y, output_grad):
        return output_grad, -output_grad * alpha

    return _backward_cpu(output_grad, x, y, func)


@BackendDispatcher.instance().register_backend_function("cpu", "mul")
def cpu_mul(x: Tensor, y: Tensor):
    shape = calculate_broadcast_shape(x.shape, y.shape)
    output_tensor = Tensor(
        dtype=x.dtype,
        shape=shape,
        device=x.device,
    )
    output_tensor._cpu_array = x._numpy() * y._numpy()
    return output_tensor


@BackendDispatcher.instance().register_backend_function("cpu", "mul_backward")
def cpu_mul_backward(output_grad: Tensor, x: Tensor, y: Tensor):
    def func(x, y, output_grad):
        return y * output_grad, x * output_grad

    return _backward_cpu(output_grad, x, y, func)


@BackendDispatcher.instance().register_backend_function("cpu", "div")
def cpu_div(x: Tensor, y: Tensor):
    shape = calculate_broadcast_shape(x.shape, y.shape)
    output_tensor = Tensor(
        dtype=x.dtype,
        shape=shape,
        device=x.device,
    )
    output_tensor._cpu_array = x._numpy() / y._numpy()
    return output_tensor


@BackendDispatcher.instance().register_backend_function("cpu", "div_backward")
def cpu_div_backward(output_grad: Tensor, x: Tensor, y: Tensor):
    def func(x, y, output_grad):
        x_grad_cpu_array = 1 / y * output_grad
        y_grad_cpu_array = -x / (y**2) * output_grad
        return x_grad_cpu_array, y_grad_cpu_array

    return _backward_cpu(output_grad, x, y, func)


@BackendDispatcher.instance().register_backend_function("cpu", "pow")
def cpu_pow(x: Tensor, y: Tensor):
    shape = calculate_broadcast_shape(x.shape, y.shape)
    output_tensor = Tensor(
        dtype=x.dtype,
        shape=shape,
        device=x.device,
    )
    output_tensor._cpu_array = np.power(x._numpy(), y._numpy())
    return output_tensor


@BackendDispatcher.instance().register_backend_function("cpu", "pow_backward")
def cpu_pow_backward(output_grad: Tensor, x: Tensor, y: Tensor):
    def func(x, y, output_grad):
        x_grad_cpu_array = y * np.power(x, y - 1) * output_grad
        y_grad_cpu_array = np.power(x, y) * np.log(x) * output_grad
        return x_grad_cpu_array, y_grad_cpu_array

    return _backward_cpu(output_grad, x, y, func)


@BackendDispatcher.instance().register_backend_function("cpu", "copy")
def cpu_copy(x: Tensor, y: Tensor):
    np.copyto(x._numpy(), y._numpy())
