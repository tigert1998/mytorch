import numpy as np

from mytorch.dtype import int8
from mytorch.backends.backend_dispatcher import BackendDispatcher


@BackendDispatcher.instance().register_backend_function("cpu", "eq")
def cpu_eq(x, y):
    from mytorch.tensor import Tensor

    output_tensor = Tensor(
        dtype=int8,
        shape=x.shape,
        device=x.device,
    )
    output_tensor.cpu_array = (x._numpy() == y._numpy()).astype(np.int8)
    return output_tensor
