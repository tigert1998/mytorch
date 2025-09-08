import numpy as np

from mytorch.backends.utils import calculate_cat_shape
from mytorch.backends.backend_dispatcher import BackendDispatcher


@BackendDispatcher.instance().register_backend_function("cpu", "cat")
def cpu_cat(tensors, dim):
    from mytorch.tensor import Tensor

    dtype = tensors[0].dtype
    device = tensors[0].device
    shape = calculate_cat_shape(tensors, dim)

    output_tensor = Tensor(
        dtype=dtype,
        shape=shape,
        device=device,
    )

    cpu_arrays = [tensor._numpy() for tensor in tensors]
    output_tensor._cpu_array = np.concatenate(cpu_arrays, axis=dim)

    return output_tensor


@BackendDispatcher.instance().register_backend_function("cpu", "cat_backward")
def cpu_cat_backward(output_grad, *args):
    from mytorch.tensor import Tensor

    tensors = args[:-1]
    dim = args[-1]

    last = 0
    splits = []
    for tensor in tensors:
        last += tensor.shape[dim]
        splits.append(last)
    splits = splits[:-1]
    arrays = np.split(output_grad._numpy(), splits, axis=dim)
    tensors_grads = [Tensor(cpu_array=array) for array in arrays]

    return tensors_grads
