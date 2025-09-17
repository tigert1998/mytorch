from typing import Optional

from mytorch.tensor import Tensor
from mytorch.autograd import DAGTracker
from mytorch.backends.backend_dispatcher import BackendDispatcher


def _batch_norm2d(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    eps: float,
    training: bool,
    momentum: float,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
):
    requires_grad = input.requires_grad
    _, channels, _, _ = input.shape
    if weight is not None and bias is not None:
        if not (
            input.dtype == weight.dtype
            and input.dtype == bias.dtype
            and input.device == weight.device
            and input.device == bias.device
            and weight.shape == (channels,)
            and bias.shape == (channels,)
        ):
            raise RuntimeError("batch_norm2d shape, dtype or device mismatch")
        requires_grad |= weight.requires_grad or bias.requires_grad

    func = BackendDispatcher.instance().dispatch(input.device.type, "batch_norm2d")
    tensor, mean, var = func(
        input, weight, bias, eps, training, momentum, running_mean, running_var
    )
    tensor.requires_grad = requires_grad and not DAGTracker.instance().no_grad

    if tensor.requires_grad:
        DAGTracker.instance().add_node(
            "batch_norm2d", [input, weight, bias, eps], [tensor], [mean, var]
        )

    return tensor


@DAGTracker.instance().register_backward_function("batch_norm2d")
def _batch_norm2d_backward(output_grad, mean, var, input, weight, bias, eps):
    func = BackendDispatcher.instance().dispatch(
        input.device.type, "batch_norm2d_backward"
    )
    return func(output_grad, mean, var, input, weight, bias, eps)
