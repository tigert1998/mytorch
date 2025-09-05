from typing import Tuple

from mytorch.backends.backend_dispatcher import BackendDispatcher


@BackendDispatcher.instance().register_backend_function("cuda", "reshape")
def cuda_reshape(x, shape: Tuple[int, ...]):
    from mytorch.tensor import Tensor

    new_x = Tensor(tensor=x)
    new_x.shape = shape
    return new_x
