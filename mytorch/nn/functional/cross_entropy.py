from mytorch.tensor import InvalidDataTypeError, Tensor
from mytorch.backends.backend_dispatcher import BackendDispatcher
from mytorch.autograd import DAGTracker
from mytorch.dtype import int64


def cross_entropy(input: Tensor, target: Tensor) -> Tensor:
    if target.dtype != int64:
        raise InvalidDataTypeError(target.dtype)
    func = BackendDispatcher.instance().dispatch(input.device.type, "cross_entropy")
    tensor = func(input, target)
    tensor.requires_grad = input.requires_grad

    if tensor.requires_grad:
        DAGTracker.instance().add_node("cross_entropy", [input, target], [tensor])

    return tensor


@DAGTracker.instance().register_backward_function("cross_entropy")
def cross_entropy_backward(output_grad: Tensor, input: Tensor, target: Tensor):
    func = BackendDispatcher.instance().dispatch(input.device.type, "cross_entropy_backward")
    return func(output_grad, input, target)
