from typing import Optional
import numpy as np

from mytorch.backends.cuda.env import CudaEnv
from mytorch.backends.backend_dispatcher import BackendDispatcher
from mytorch.tensor import Tensor


@BackendDispatcher.instance().register_backend_function("cuda", "conv2d")
def cuda_conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor], stride, padding):
    output_shape = [
        input.shape[0],
        weight.shape[0],
        *[
            (input.shape[2 + i] + padding[i] * 2 - weight.shape[2 + i]) // stride[i] + 1
            for i in range(2)
        ],
    ]
    output_tensor = Tensor(dtype=input.dtype, shape=output_shape, device=input.device)

    func_name = f"ConvForwardImplicitGEMM_{input.dtype.name}"
    stream = CudaEnv.instance().kernel_and_stream_manager.get_stream(input.device.index)
    CudaEnv.instance().library.run(
        func_name,
        [
            np.array(input.shape[0], np.int32),
            np.array(input.shape[1], np.int32),
            np.array(input.shape[2], np.int32),
            np.array(input.shape[3], np.int32),
            np.array(weight.shape[2], np.int32),
            np.array(weight.shape[3], np.int32),
            np.array(stride[0], np.int32),
            np.array(stride[1], np.int32),
            np.array(padding[0], np.int32),
            np.array(padding[1], np.int32),
            np.array(weight.shape[0], np.int32),
            input,
            weight,
            bias,
            output_tensor,
        ],
        stream,
    )

    return output_tensor


@BackendDispatcher.instance().register_backend_function("cuda", "conv2d_backward")
def cuda_conv2d_backward(
    output_grad: Tensor,
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride,
    padding,
):
    input_grad = Tensor(shape=input.shape, device=input.device, dtype=input.dtype)
    weight_grad = Tensor(shape=weight.shape, device=weight.device, dtype=weight.dtype)

    func_name = f"ConvBackwardImplicitGEMM_{input.dtype.name}"
    stream = CudaEnv.instance().kernel_and_stream_manager.get_stream(input.device.index)
    CudaEnv.instance().library.run(
        func_name,
        [
            np.array(input.shape[0], np.int32),
            np.array(input.shape[1], np.int32),
            np.array(input.shape[2], np.int32),
            np.array(input.shape[3], np.int32),
            np.array(weight.shape[2], np.int32),
            np.array(weight.shape[3], np.int32),
            np.array(stride[0], np.int32),
            np.array(stride[1], np.int32),
            np.array(padding[0], np.int32),
            np.array(padding[1], np.int32),
            np.array(weight.shape[0], np.int32),
            input,
            weight,
            output_grad,
            input_grad,
            weight_grad,
        ],
        stream,
    )

    if bias is not None:
        bias_grad = output_grad.sum((0, 2, 3))
    else:
        bias_grad = None

    return [input_grad, weight_grad, bias_grad]
