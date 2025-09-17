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

    func_name = f"batch_norm2d_reference_{input.dtype.name}"
    cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
    cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
        "batch_norm.cu", func_name, input.device.index
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

    cuda_kernel.run(
        (1, 1, 1),
        (1, 1, 1),
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
    cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
    cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
        "batch_norm.cu", func_name, input.device.index
    )

    cuda_kernel.run(
        (1, 1, 1),
        (1, 1, 1),
        [
            np.array(batch_size),
            np.array(channels),
            np.array(height),
            np.array(width),
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
    )

    if weight is None and bias is None:
        weight_grad = bias_grad = None

    return [input_grad, weight_grad, bias_grad]
