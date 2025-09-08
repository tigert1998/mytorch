import numpy as np

from mytorch.backends.utils import calculate_reduce_shape
from mytorch.backends.backend_dispatcher import BackendDispatcher


@BackendDispatcher.instance().register_backend_function("cpu", "sum_scale")
def cpu_sum_scale(tensor, dim, keepdim, scale):
    from mytorch.tensor import Tensor

    output_shape = calculate_reduce_shape(tensor.shape, dim, keepdim)
    output_tensor = Tensor(shape=output_shape, device=tensor.device, dtype=tensor.dtype)
    output_tensor.cpu_array = (
        np.sum(tensor._numpy(), axis=dim, keepdims=keepdim) * scale
    )
    return output_tensor


@BackendDispatcher.instance().register_backend_function("cpu", "sum_scale_backward")
def cpu_sum_scale_backward(output_grad, tensor, dim, keepdim, scale):
    from mytorch.tensor import Tensor

    input_grad = Tensor(shape=tensor.shape, device=tensor.device, dtype=tensor.dtype)

    if keepdim:
        input_grad.cpu_array = output_grad._numpy() * scale
    else:
        input_grad.cpu_array = np.expand_dims(output_grad._numpy(), dim) * scale

    return [input_grad]
