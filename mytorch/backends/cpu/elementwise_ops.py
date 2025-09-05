import numpy as np

from mytorch.backends.backend_dispatcher import BackendDispatcher


@BackendDispatcher.instance().register_backend_function("cpu", "fill")
def _fill(x, value):
    np.copyto(x._numpy(), value)


@BackendDispatcher.instance().register_backend_function("cpu", "normal")
def _normal(x, seed, mean, stddev):
    np.random.seed(seed)
    x.cpu_array = np.random.normal(mean, stddev, x.shape).astype(x.dtype.np_dtype)


@BackendDispatcher.instance().register_backend_function("cpu", "uniform")
def _uniform(x, seed, a, b):
    np.random.seed(seed)
    x.cpu_array = np.random.uniform(low=a, high=b, size=x.shape).astype(
        x.dtype.np_dtype
    )


@BackendDispatcher.instance().register_backend_function("cpu", "relu")
def _relu(x):
    from mytorch.tensor import Tensor

    new_x = Tensor(device=x.device, shape=x.shape, dtype=x.dtype)
    new_x.cpu_array = np.maximum(x._numpy(), 0)
    return new_x


@BackendDispatcher.instance().register_backend_function("cpu", "relu_backward")
def _relu_backward(output_grad, x):
    from mytorch.tensor import Tensor

    x_grad = Tensor(device=x.device, shape=x.shape, dtype=x.dtype)
    x_grad.cpu_array = output_grad._numpy().copy()
    x_grad.cpu_array[x._numpy() < 0] = 0
    return [x_grad]
