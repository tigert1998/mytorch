from mytorch.autograd import DAGTracker
from mytorch.tensor import (
    Tensor,
    MismatchDevicesError,
    MismatchDataTypesError,
)
from mytorch.backends.backend_dispatcher import BackendDispatcher


def conv2d(input: Tensor, weight: Tensor, bias=None, stride=1, padding=0) -> Tensor:
    if not (
        input.device == weight.device and (bias is None or input.device == bias.device)
    ):
        devices = [input.device, weight.device]
        devices += [bias.device] if bias is not None else []
        raise MismatchDevicesError(devices)
    if not (
        input.dtype == weight.dtype and (bias is None or input.dtype == bias.dtype)
    ):
        dtypes = [input.dtype, weight.dtype]
        dtypes += [bias.dtype] if bias is not None else []
        raise MismatchDataTypesError(dtypes)

    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding

    func = BackendDispatcher.instance().dispatch(input.device.type, "conv2d")
    tensor = func(input, weight, bias, stride, padding)
    tensor.requires_grad = (
        input.requires_grad
        or weight.requires_grad
        or (bias is not None and bias.requires_grad)
    ) and not DAGTracker.instance().no_grad

    if tensor.requires_grad:
        DAGTracker.instance().add_node(
            "conv2d", [input, weight, bias, stride, padding], [tensor]
        )

    return tensor


@DAGTracker.instance().register_backward_function("conv2d")
def conv2d_backward(output_grad, input, weight, bias, stride, padding):
    func = BackendDispatcher.instance().dispatch(input.device.type, "conv2d_backward")
    input_grad, weight_grad, bias_grad = func(
        output_grad, input, weight, bias, stride, padding
    )
    return [input_grad, weight_grad, bias_grad]
