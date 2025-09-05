from typing import Tuple

from mytorch.backends.backend_dispatcher import BackendDispatcher


@BackendDispatcher.instance().register_backend_function("cpu", "reshape")
def cpu_reshape(x, shape: Tuple[int, ...]):
    from mytorch.tensor import Tensor

    new_x = Tensor(tensor=x)
    new_x.shape = shape
    new_x.cpu_array = new_x._numpy().reshape(new_x.shape)
    return new_x
