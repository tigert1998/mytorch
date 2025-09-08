import numpy as np

from mytorch.backends.backend_dispatcher import BackendDispatcher


@BackendDispatcher.instance().register_backend_function("cpu", "fill")
def cpu_fill(x, value):
    np.copyto(x._numpy(), value)


@BackendDispatcher.instance().register_backend_function("cpu", "normal")
def cpu_normal(x, seed, mean, stddev):
    np.random.seed(seed)
    x._cpu_array = np.random.normal(mean, stddev, x.shape).astype(x.dtype.np_dtype)


@BackendDispatcher.instance().register_backend_function("cpu", "uniform")
def cpu_uniform(x, seed, a, b):
    np.random.seed(seed)
    x._cpu_array = np.random.uniform(low=a, high=b, size=x.shape).astype(
        x.dtype.np_dtype
    )


@BackendDispatcher.instance().register_backend_function("cpu", "relu")
def cpu_relu(x):
    from mytorch.tensor import Tensor

    new_x = Tensor(device=x.device, shape=x.shape, dtype=x.dtype)
    new_x._cpu_array = np.maximum(x._numpy(), 0)
    return new_x


@BackendDispatcher.instance().register_backend_function("cpu", "relu_backward")
def cpu_relu_backward(output_grad, x):
    from mytorch.tensor import Tensor

    x_grad = Tensor(device=x.device, shape=x.shape, dtype=x.dtype)
    x_grad._cpu_array = output_grad._numpy().copy()
    x_grad._cpu_array[x._numpy() < 0] = 0
    return [x_grad]
