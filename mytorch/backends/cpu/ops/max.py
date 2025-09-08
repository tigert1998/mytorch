import numpy as np

from mytorch.backends.backend_dispatcher import BackendDispatcher


@BackendDispatcher.instance().register_backend_function("cpu", "max")
def cpu_max(tensor, dim, keepdim):
    from mytorch.tensor import Tensor

    output_cpu_array = np.max(tensor._numpy(), axis=dim, keepdims=keepdim).astype(
        tensor.dtype.np_dtype
    )
    output = Tensor(output_cpu_array, device=tensor.device)
    if dim is not None:
        indices_cpu_array = np.argmax(
            tensor._numpy(), axis=dim, keepdims=keepdim
        ).astype(np.int64)
        indices = Tensor(indices_cpu_array, device=tensor.device)
        return output, indices
    else:
        return output
