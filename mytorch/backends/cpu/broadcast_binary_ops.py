import numpy as np
import numpy.typing as npt
from typing import Tuple, List

from mytorch.backends.utils import calculate_broadcast_shape
from mytorch.backends.backend_dispatcher import BackendDispatcher


def _tile_tensor_cpu(x, output_grad_shape) -> Tuple[npt.NDArray, List[int]]:
    x_shape = (1,) * (len(output_grad_shape) - len(x.shape)) + x.shape
    x_axis = [i for i in range(len(x_shape)) if x_shape[i] < output_grad_shape[i]]
    x_tile_reps = [
        output_grad_shape[i] if x_shape[i] < output_grad_shape[i] else 1
        for i in range(len(x_shape))
    ]
    x_tile = np.tile(x, x_tile_reps)
    return x_tile, x_axis


def _backward_cpu(output_grad, x, y, func):
    from mytorch.tensor import Tensor

    x_grad = Tensor(dtype=x.dtype, shape=x.shape, device=x.device)
    x_grad.fill_(0)
    y_grad = Tensor(dtype=y.dtype, shape=y.shape, device=y.device)
    y_grad.fill_(0)

    x_tile, x_axis = _tile_tensor_cpu(x._numpy(), output_grad.shape)
    y_tile, y_axis = _tile_tensor_cpu(y._numpy(), output_grad.shape)
    x_grad_cpu_array, y_grad_cpu_array = func(x_tile, y_tile, output_grad._numpy())
    x_grad.cpu_array = x_grad_cpu_array.sum(axis=tuple(x_axis)).reshape(x.shape)
    y_grad.cpu_array = y_grad_cpu_array.sum(axis=tuple(y_axis)).reshape(y.shape)

    return x_grad, y_grad


@BackendDispatcher.instance().register_backend_function("cpu", "add")
def add(x, y, alpha):
    from mytorch.tensor import Tensor

    shape = calculate_broadcast_shape(x.shape, y.shape)
    output_tensor = Tensor(
        dtype=x.dtype,
        shape=shape,
        device=x.device,
    )
    output_tensor.cpu_array = x._numpy() + y._numpy() * alpha
    return output_tensor


@BackendDispatcher.instance().register_backend_function("cpu", "add_backward")
def add_backward(output_grad, x, y, alpha):
    def func(x, y, output_grad):
        return output_grad, output_grad * alpha

    return _backward_cpu(output_grad, x, y, func)


@BackendDispatcher.instance().register_backend_function("cpu", "sub")
def sub(x, y, alpha):
    from mytorch.tensor import Tensor

    shape = calculate_broadcast_shape(x.shape, y.shape)
    output_tensor = Tensor(
        dtype=x.dtype,
        shape=shape,
        device=x.device,
    )
    output_tensor.cpu_array = x._numpy() - y._numpy() * alpha
    return output_tensor


@BackendDispatcher.instance().register_backend_function("cpu", "sub_backward")
def sub_backward(output_grad, x, y, alpha):
    def func(x, y, output_grad):
        return output_grad, -output_grad * alpha

    return _backward_cpu(output_grad, x, y, func)


@BackendDispatcher.instance().register_backend_function("cpu", "mul")
def mul(x, y):
    from mytorch.tensor import Tensor

    shape = calculate_broadcast_shape(x.shape, y.shape)
    output_tensor = Tensor(
        dtype=x.dtype,
        shape=shape,
        device=x.device,
    )
    output_tensor.cpu_array = x._numpy() * y._numpy()
    return output_tensor


@BackendDispatcher.instance().register_backend_function("cpu", "mul_backward")
def mul_backward(output_grad, x, y):
    def func(x, y, output_grad):
        return y * output_grad, x * output_grad

    return _backward_cpu(output_grad, x, y, func)


@BackendDispatcher.instance().register_backend_function("cpu", "div")
def div(x, y):
    from mytorch.tensor import Tensor

    shape = calculate_broadcast_shape(x.shape, y.shape)
    output_tensor = Tensor(
        dtype=x.dtype,
        shape=shape,
        device=x.device,
    )
    output_tensor.cpu_array = x._numpy() * y._numpy()
    return output_tensor


@BackendDispatcher.instance().register_backend_function("cpu", "div_backward")
def div_backward(output_grad, x, y):
    def func(x, y, output_grad):
        x_grad_cpu_array = 1 / y * output_grad
        y_grad_cpu_array = -x / (y**2) * output_grad
        return x_grad_cpu_array, y_grad_cpu_array

    return _backward_cpu(output_grad, x, y, func)


@BackendDispatcher.instance().register_backend_function("cpu", "pow")
def pow(x, y):
    from mytorch.tensor import Tensor

    shape = calculate_broadcast_shape(x.shape, y.shape)
    output_tensor = Tensor(
        dtype=x.dtype,
        shape=shape,
        device=x.device,
    )
    output_tensor.cpu_array = np.power(x._numpy(), y._numpy())
    return output_tensor


@BackendDispatcher.instance().register_backend_function("cpu", "pow_backward")
def pow_backward(output_grad, x, y):
    def func(x, y, output_grad):
        x_grad_cpu_array = y * np.power(x, y - 1) * output_grad
        y_grad_cpu_array = np.power(x, y) * np.log(x) * output_grad
        return x_grad_cpu_array, y_grad_cpu_array

    return _backward_cpu(output_grad, x, y, func)


@BackendDispatcher.instance().register_backend_function("cpu", "copy")
def copy(x, y):
    np.copyto(x.cpu_array, y.cpu_array)
