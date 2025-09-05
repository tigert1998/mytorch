from functools import cache

import numpy as np

from mytorch.tensor import Tensor, shape_size, InvalidDeviceError
from mytorch.backends.cuda.env import CudaEnv
from mytorch.autograd import DAGTracker
from mytorch.dtype import DType
from mytorch.backends.backend_dispatcher import BackendDispatcher


def _cast(x: Tensor, dtype: DType):
    if x.dtype == dtype:
        return x

    func = BackendDispatcher.instance().dispatch(x.device.type, "cast")
    output_tensor = func(x, dtype)
    output_tensor.requires_grad = (
        x.requires_grad and x.dtype.is_floating and dtype.is_floating
    )

    if output_tensor.requires_grad:
        DAGTracker.instance().add_node("cast", [x, dtype], [output_tensor])

    return output_tensor


@DAGTracker.instance().register_backward_function("cast")
def _cast_backward(output_grad: Tensor, x: Tensor, dtype: DType):
    func = BackendDispatcher.instance().dispatch(x.device.type, "cast")
    x_grad = func(output_grad, x.dtype)
    return [x_grad]
