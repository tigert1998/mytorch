from mytorch.tensor import (
    MismatchDevicesError,
    Tensor,
)
from mytorch.autograd import DAGTracker
from mytorch.backends.backend_dispatcher import BackendDispatcher


def add(x: Tensor, y: Tensor, alpha) -> Tensor:
    if x.device != y.device:
        raise MismatchDevicesError([x.device, y.device])

    func = BackendDispatcher.instance().dispatch(x.device.type, "add")
    z = func(x, y, alpha)
    z.requires_grad = x.requires_grad or y.requires_grad

    if z.requires_grad:
        DAGTracker.instance().add_node("add", [x, y, alpha], [z])

    return z


@DAGTracker.instance().register_backward_function("add")
def add_backward(output_tensor, x: Tensor, y: Tensor, alpha) -> Tensor:
    func = BackendDispatcher.instance().dispatch(x.device.type, "add_backward")
    return func(output_tensor, x, y, alpha)


def sub(x: Tensor, y: Tensor, alpha) -> Tensor:
    if x.device != y.device:
        raise MismatchDevicesError([x.device, y.device])

    func = BackendDispatcher.instance().dispatch(x.device.type, "sub")
    z = func(x, y, alpha)
    z.requires_grad = x.requires_grad or y.requires_grad

    if z.requires_grad:
        DAGTracker.instance().add_node("sub", [x, y, alpha], [z])

    return z


@DAGTracker.instance().register_backward_function("sub")
def sub_backward(output_tensor, x: Tensor, y: Tensor, alpha) -> Tensor:
    func = BackendDispatcher.instance().dispatch(x.device.type, "sub_backward")
    return func(output_tensor, x, y, alpha)


def mul(x: Tensor, y: Tensor) -> Tensor:
    if x.device != y.device:
        raise MismatchDevicesError([x.device, y.device])

    func = BackendDispatcher.instance().dispatch(x.device.type, "mul")
    z = func(x, y)
    z.requires_grad = x.requires_grad or y.requires_grad

    if z.requires_grad:
        DAGTracker.instance().add_node("mul", [x, y], [z])

    return z


@DAGTracker.instance().register_backward_function("mul")
def mul_backward(output_tensor, x: Tensor, y: Tensor) -> Tensor:
    func = BackendDispatcher.instance().dispatch(x.device.type, "mul_backward")
    return func(output_tensor, x, y)


def div(x: Tensor, y: Tensor) -> Tensor:
    if x.device != y.device:
        raise MismatchDevicesError([x.device, y.device])

    func = BackendDispatcher.instance().dispatch(x.device.type, "div")
    z = func(x, y)
    z.requires_grad = x.requires_grad or y.requires_grad

    if z.requires_grad:
        DAGTracker.instance().add_node("div", [x, y], [z])

    return z


@DAGTracker.instance().register_backward_function("div")
def div_backward(output_tensor, x: Tensor, y: Tensor) -> Tensor:
    func = BackendDispatcher.instance().dispatch(x.device.type, "div_backward")
    return func(output_tensor, x, y)


def pow(x: Tensor, y: Tensor) -> Tensor:
    if x.device != y.device:
        raise MismatchDevicesError([x.device, y.device])

    func = BackendDispatcher.instance().dispatch(x.device.type, "pow")
    z = func(x, y)
    z.requires_grad = x.requires_grad or y.requires_grad

    if z.requires_grad:
        DAGTracker.instance().add_node("pow", [x, y], [z])

    return z


@DAGTracker.instance().register_backward_function("pow")
def pow_backward(output_tensor, x: Tensor, y: Tensor) -> Tensor:
    func = BackendDispatcher.instance().dispatch(x.device.type, "pow_backward")
    return func(output_tensor, x, y)


def _copy(x: Tensor, y: Tensor):
    if x.device != y.device:
        raise MismatchDevicesError([x.device, y.device])
    func = BackendDispatcher.instance().dispatch(x.device.type, "copy")
    func(x, y)
