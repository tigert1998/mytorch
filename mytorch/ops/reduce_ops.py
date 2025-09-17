from typing import Tuple

from mytorch.tensor import shape_size, Tensor
from mytorch.autograd import DAGTracker
from mytorch.backends.backend_dispatcher import BackendDispatcher
from mytorch.ops.elementwise_ops import sqrt, sqr


def _sum_scale(x: Tensor, dim=None, keepdim=False, scale=1) -> Tensor:
    if dim is None:
        dim = tuple(range(len(x.shape)))
    elif isinstance(dim, int):
        dim = (dim,)
    func = BackendDispatcher.instance().dispatch(x.device.type, "sum_scale")
    output_tensor = func(x, dim, keepdim, scale)
    output_tensor.requires_grad = x.requires_grad and not DAGTracker.instance().no_grad
    if output_tensor.requires_grad:
        DAGTracker.instance().add_node(
            "sum_scale", [x, dim, keepdim, scale], [output_tensor]
        )
    return output_tensor


@DAGTracker.instance().register_backward_function("sum_scale")
def _sum_scale_backward(
    output_grad: Tensor, x: Tensor, dim=None, keepdim=False, scale=1
) -> Tuple[Tensor, ...]:
    func = BackendDispatcher.instance().dispatch(x.device.type, "sum_scale_backward")
    return func(output_grad, x, dim, keepdim, scale)


def sum(x: Tensor, dim=None, keepdim=False) -> Tensor:
    return _sum_scale(x, dim, keepdim, 1)


def mean(x: Tensor, dim=None, keepdim=False) -> Tensor:
    if dim is None:
        dim = tuple(range(len(x.shape)))
    elif isinstance(dim, int):
        dim = (dim,)
    scale = 1 / shape_size([x.shape[i] for i in dim])
    return _sum_scale(x, dim, keepdim, scale)


def var(x: Tensor, dim=None, correction=1, keepdim=False) -> Tensor:
    if dim is None:
        dim = tuple(range(len(x.shape)))
    elif isinstance(dim, int):
        dim = (dim,)
    scale = 1 / (shape_size([x.shape[i] for i in dim]) - correction)
    x_minus_mean = x - mean(x, dim=dim, keepdim=True)
    return _sum_scale(sqr(x_minus_mean), dim, keepdim, scale)


def std(x: Tensor, dim=None, correction=1, keepdim=False) -> Tensor:
    return sqrt(var(x, dim, correction, keepdim))
