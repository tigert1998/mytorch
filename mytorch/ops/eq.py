from mytorch.tensor import Tensor, MismatchDataTypesError
from mytorch.backends.backend_dispatcher import BackendDispatcher


def eq(x: Tensor, y: Tensor):
    if x.dtype != y.dtype:
        raise MismatchDataTypesError([x.dtype, y.dtype])
    func = BackendDispatcher().instance().dispatch(x.device.type, "eq")
    output_tensor = func(x, y)
    return output_tensor
