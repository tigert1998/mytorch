from typing import Optional

import numpy as np

from mytorch.tensor import Tensor
from mytorch.backends.cuda.env import CudaEnv
from mytorch.backends.backend_dispatcher import BackendDispatcher


@BackendDispatcher.instance().register_backend_function("cuda", "batch_norm2d")
def cuda_batch_norm2d(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    eps: float,
    training: bool,
    momentum: float,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
):
    batch_size, channels, height, width = input.shape

    tensor = Tensor(
        dtype=input.dtype,
        shape=input.shape,
        device=input.device,
    )

    mean = Tensor(
        dtype=input.dtype,
        shape=(channels,),
        device=input.device,
    )
    var = Tensor(
        dtype=input.dtype,
        shape=(channels,),
        device=input.device,
    )

    func_name = f"batch_norm2d_reference_{input.dtype.name}"
    stream = CudaEnv.instance().kernel_and_stream_manager.get_stream(input.device.index)
    CudaEnv.instance().library.run(
        func_name,
        [
            np.array(batch_size),
            np.array(channels),
            np.array(height),
            np.array(width),
            input,
            mean,
            var,
            np.array(eps, dtype=input.dtype.np_dtype),
            weight,
            bias,
            np.array(training, dtype=np.int8),
            np.array(momentum, dtype=input.dtype.np_dtype),
            running_mean,
            running_var,
            tensor,
        ],
        stream,
    )

    return tensor, mean, var


@BackendDispatcher.instance().register_backend_function("cuda", "batch_norm2d_backward")
def cuda_batch_norm2d_backward(output_grad, mean, var, input, weight, bias, eps):
    batch_size, channels, height, width = input.shape

    input_grad = Tensor(
        dtype=input.dtype,
        shape=input.shape,
        device=input.device,
    )
    weight_grad = Tensor(dtype=input.dtype, shape=(channels,), device=input.device)
    bias_grad = Tensor(dtype=input.dtype, shape=(channels,), device=input.device)

    func_name = f"batch_norm2d_backward_reference_{input.dtype.name}"
    stream = CudaEnv.instance().kernel_and_stream_manager.get_stream(input.device.index)
    CudaEnv.instance().library.run(
        func_name,
        [
            np.array(batch_size, dtype=np.int32),
            np.array(channels, dtype=np.int32),
            np.array(height, dtype=np.int32),
            np.array(width, dtype=np.int32),
            input,
            mean,
            var,
            np.array(eps, dtype=np.float32),
            weight,
            bias,
            input_grad,
            weight_grad,
            bias_grad,
            output_grad,
        ],
        stream,
    )

    if weight is None and bias is None:
        weight_grad = bias_grad = None

    return [input_grad, weight_grad, bias_grad]
