from typing import Tuple

from mytorch.tensor import (
    MismatchDevicesError,
    shape_size,
    Tensor,
)
from mytorch.autograd import DAGTracker
from mytorch.backends.backend_dispatcher import BackendDispatcher


def mm(x: Tensor, y: Tensor) -> Tensor:
    if x.device != y.device:
        raise MismatchDevicesError([x.device, y.device])

    func = BackendDispatcher.instance().dispatch(x.device.type, "mm")
    z = func(x, y)
    z.requires_grad = x.requires_grad or y.requires_grad

    if z.requires_grad:
        DAGTracker.instance().add_node("mm", [x, y], [z])

    return z


@DAGTracker.instance().register_backward_function("mm")
def mm_backward(output_grad: Tensor, x: Tensor, y: Tensor):
    if not (output_grad.device == x.device and output_grad.device == y.device):
        raise MismatchDevicesError([output_grad.device, x.device, y.device])

    func = BackendDispatcher.instance().dispatch(x.device.type, "mm_backward")
    x_grad, y_grad = func(output_grad, x, y)

    return [x_grad, y_grad]


def bmm(x: Tensor, y: Tensor) -> Tensor:
    if x.device != y.device:
        raise MismatchDevicesError([x.device, y.device])

    func = BackendDispatcher.instance().dispatch(x.device.type, "bmm")
    z = func(x, y)
    z.requires_grad = x.requires_grad or y.requires_grad

    if z.requires_grad:
        DAGTracker.instance().add_node("bmm", [x, y], [z])

    return z


@DAGTracker.instance().register_backward_function("bmm")
def bmm_backward(output_grad: Tensor, x: Tensor, y: Tensor):
    if not (output_grad.device == x.device and output_grad.device == y.device):
        raise MismatchDevicesError([output_grad.device, x.device, y.device])

    func = BackendDispatcher.instance().dispatch(x.device.type, "bmm_backward")
    x_grad, y_grad = func(x, y)

    return [x_grad, y_grad]


def permute(x: Tensor, dims: Tuple[int, ...]) -> Tensor:
    if len(dims) != len(x.shape):
        raise RuntimeError(f"permute dims is invalid: {dims}")
    dims = tuple([(i + len(dims) if i < 0 else i) for i in dims])

    func = BackendDispatcher.instance().dispatch(x.device.type, "permute")
    output_tensor = func(x, dims)
    output_tensor.requires_grad = x.requires_grad

    if output_tensor.requires_grad:
        DAGTracker.instance().add_node("permute", [x, dims], [output_tensor])

    return output_tensor


@DAGTracker.instance().register_backward_function("permute")
def permute_backward(output_grad: Tensor, x: Tensor, dims: Tuple[int, ...]):
    func = BackendDispatcher.instance().dispatch(x.device.type, "permute_backward")
    (input_grad,) = func(output_grad, x, dims)
    return [input_grad]


def reshape(x: Tensor, shape: Tuple[int, ...]) -> Tensor:
    func = BackendDispatcher.instance().dispatch(x.device.type, "reshape")
    output_tensor = func(x, shape)
    output_tensor.requires_grad = x.requires_grad

    if output_tensor.requires_grad:
        DAGTracker.instance().add_node(
            "reshape", [x, output_tensor.shape], [output_tensor]
        )

    return output_tensor


@DAGTracker.instance().register_backward_function("reshape")
def reshape_backward(output_grad: Tensor, x: Tensor, shape: Tuple[int, ...]):
    input_grad = Tensor(tensor=output_grad)
    input_grad.shape = x.shape

    return [input_grad]
