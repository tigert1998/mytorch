import numpy as np

from mytorch.backends.backend_dispatcher import BackendDispatcher


@BackendDispatcher.instance().register_backend_function("cpu", "mm")
def cpu_mm(x, y):
    from mytorch.tensor import Tensor

    z_cpu_array = np.matmul(x._numpy(), y._numpy())
    return Tensor(cpu_array=z_cpu_array, device="cpu")


@BackendDispatcher.instance().register_backend_function("cpu", "mm_backward")
def cpu_mm_backward(output_grad, x, y):
    from mytorch.tensor import Tensor

    x_grad_cpu_array = np.matmul(output_grad._numpy(), y._numpy().T)
    y_grad_cpu_array = np.matmul(x._numpy().T, output_grad._numpy())
    x_grad = Tensor(cpu_array=x_grad_cpu_array, device="cpu")
    y_grad = Tensor(cpu_array=y_grad_cpu_array, device="cpu")
    return x_grad, y_grad


@BackendDispatcher.instance().register_backend_function("cpu", "bmm")
def cpu_bmm(x, y):
    from mytorch.tensor import Tensor

    z_cpu_array = np.matmul(x._numpy(), y._numpy())
    return Tensor(cpu_array=z_cpu_array, device="cpu")


@BackendDispatcher.instance().register_backend_function("cpu", "bmm_backward")
def cpu_bmm_backward(output_grad, x, y):
    from mytorch.tensor import Tensor

    x_grad_cpu_array = np.matmul(
        output_grad._numpy(), np.transpose(y._numpy(), (0, 2, 1))
    )
    y_grad_cpu_array = np.matmul(
        np.transpose(x._numpy(), (0, 2, 1)), output_grad._numpy()
    )
    x_grad = Tensor(cpu_array=x_grad_cpu_array, device="cpu")
    y_grad = Tensor(cpu_array=y_grad_cpu_array, device="cpu")
    return x_grad, y_grad
