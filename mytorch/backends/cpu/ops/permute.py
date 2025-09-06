from typing import Tuple

import numpy as np


from mytorch.backends.backend_dispatcher import BackendDispatcher


@BackendDispatcher.instance().register_backend_function("cpu", "permute")
def cpu_permute(x, dims: Tuple[int, ...]):
    from mytorch.tensor import Tensor

    return Tensor(
        cpu_array=np.transpose(x._numpy(), dims),
        dtype=x.dtype,
        device=x.device,
    )


@BackendDispatcher.instance().register_backend_function("cpu", "permute_backward")
def cpu_permute_backward(output_grad, x, dims: Tuple[int, ...]):
    from mytorch.tensor import Tensor

    reverse_dims = [-1 for _ in range(len(dims))]
    # [1, 2, 0] => [2, 0, 1]
    for i, permute in enumerate(dims):
        reverse_dims[permute] = i
    for i in reverse_dims:
        if i < 0:
            raise RuntimeError(f"Invalid dimension {dims} in CPU backward pass")
    return [
        Tensor(
            cpu_array=np.transpose(output_grad._numpy(), tuple(reverse_dims)),
            dtype=x.dtype,
            device=x.device,
            shape=x.shape,
        )
    ]
