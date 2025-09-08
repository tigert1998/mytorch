from mytorch.tensor import (
    MismatchDevicesError,
    Tensor,
)
from mytorch.autograd import DAGTracker
from mytorch.backends.backend_dispatcher import BackendDispatcher


def _fill(x: Tensor, value):
    func = BackendDispatcher.instance().dispatch(x.device.type, "fill")
    func(x, value)


def _normal(x: Tensor, seed, mean, stddev):
    func = BackendDispatcher.instance().dispatch(x.device.type, "normal")
    func(x, seed, mean, stddev)


def _uniform(x: Tensor, seed, a, b):
    func = BackendDispatcher.instance().dispatch(x.device.type, "uniform")
    func(x, seed, a, b)


def _relu(x: Tensor) -> Tensor:
    func = BackendDispatcher.instance().dispatch(x.device.type, "relu")
    output_tensor = func(x)
    output_tensor.requires_grad = x.requires_grad
    if output_tensor.requires_grad:
        DAGTracker.instance().add_node("relu", [x], [output_tensor])
    return output_tensor


@DAGTracker.instance().register_backward_function("relu")
def relu_backward(output_grad, x):
    func = BackendDispatcher.instance().dispatch(
        output_grad.device.type, "relu_backward"
    )
    return func(output_grad, x)


def sqr(x: Tensor) -> Tensor:
    func = BackendDispatcher.instance().dispatch(x.device.type, "sqr")
    output_tensor = func(x)
    output_tensor.requires_grad = x.requires_grad
    if output_tensor.requires_grad:
        DAGTracker.instance().add_node("sqr", [x], [output_tensor])
    return output_tensor


@DAGTracker.instance().register_backward_function("sqr")
def sqr_backward(output_grad, x):
    func = BackendDispatcher.instance().dispatch(
        output_grad.device.type, "sqr_backward"
    )
    return func(output_grad, x)


def sqrt(x: Tensor) -> Tensor:
    func = BackendDispatcher.instance().dispatch(x.device.type, "sqrt")
    output_tensor = func(x)
    output_tensor.requires_grad = x.requires_grad
    if output_tensor.requires_grad:
        DAGTracker.instance().add_node("sqrt", [x], [output_tensor])
    return output_tensor


@DAGTracker.instance().register_backward_function("sqrt")
def sqrt_backward(output_grad, x):
    func = BackendDispatcher.instance().dispatch(
        output_grad.device.type, "sqrt_backward"
    )
    return func(output_grad, x)
