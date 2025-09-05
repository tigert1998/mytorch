from typing import Tuple

from mytorch.backends.backend_dispatcher import BackendDispatcher
from mytorch.backends.utils import calculate_reshaped_shape


@BackendDispatcher.instance().register_backend_function("cuda", "reshape")
def cuda_reshape(x, shape: Tuple[int, ...]):
    from mytorch.tensor import Tensor

    new_x = Tensor(tensor=x)
    new_x.shape = calculate_reshaped_shape(x.shape, shape)
    return new_x
