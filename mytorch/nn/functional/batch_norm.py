import numpy as np

from mytorch.tensor import InvalidDataTypeError, InvalidDeviceError, Tensor
from mytorch.cuda.env import CudaEnv
from mytorch.autograd import DAGTracker


def _batch_norm_2d(
    input, weight, bias, eps, training, momentum, running_mean, running_var
):
    requires_grad = input.requires_grad
    batch_size, channels, height, width = input.shape
    if weight is not None and bias is not None:
        if not (
            input.dtype == weight.dtype
            and input.dtype == bias.dtype
            and input.device == weight.device
            and input.device == bias.device
            and weight.shape == (channels,)
            and bias.shape == (channels,)
        ):
            raise RuntimeError("BatchNorm2d shape/dtype/device mismatch")
        requires_grad |= weight.requires_grad or bias.requires_grad

    tensor = Tensor(
        dtype=input.dtype,
        shape=input.shape,
        device=input.device,
        requires_grad=requires_grad,
    )

    if input.device.type == "cuda":
        if input.dtype == np.float32:
            func_name = "batch_norm2d_reference_fp32"
        elif input.dtype == np.float16:
            func_name = "batch_norm2d_reference_fp16"
        else:
            raise InvalidDataTypeError(input.dtype)
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
                np.array(eps, dtype=input.dtype),
                weight,
                bias,
                np.array(training, dtype=np.int8),
                np.array(momentum, dtype=input.dtype),
                running_mean,
                running_var,
                tensor,
            ],
        )

    elif input.device.type == "cpu":
        raise NotImplementedError()

    else:
        raise InvalidDeviceError(input.device.type)

    if requires_grad:
        DAGTracker.instance().add_node(
            "batch_norm_2d", [input, weight, bias, eps], [tensor], [mean, var]
        )

    return tensor


@DAGTracker.instance().register_backward_function("batch_norm_2d")
def _batch_norm_2d_backward(output_grad, mean, var, input, weight, bias, eps):
    batch_size, channels, height, width = input.shape

    input_grad = Tensor(
        dtype=input.dtype,
        shape=input.shape,
        device=input.device,
    )
    if weight is not None and bias is not None:
        weight_grad = Tensor(
            dtype=weight.dtype, shape=weight.shape, device=weight.device
        )
        bias_grad = Tensor(dtype=bias.dtype, shape=bias.shape, device=bias.device)
    else:
        weight_grad = None
        bias_grad = None

    if input.device.type == "cuda":
        if input.dtype == np.float32:
            func_name = "batch_norm2d_backward_reference_fp32"
        elif input.dtype == np.float16:
            func_name = "batch_norm2d_backward_reference_fp16"
        else:
            raise InvalidDataTypeError(input.dtype)
        cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
        cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
            "batch_norm.cu", func_name, input.device.index
        )

        mean_grad = Tensor(
            dtype=input.dtype,
            shape=(channels,),
            device=input.device,
        )
        var_grad = Tensor(
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
                np.array(eps, dtype=np.float32),
                weight,
                bias,
                input_grad,
                mean_grad,
                var_grad,
                weight_grad,
                bias_grad,
                output_grad,
            ],
        )

    elif input.device.type == "cpu":
        raise NotImplementedError()

    else:
        raise InvalidDeviceError(input.device.type)

    return [input_grad, weight_grad, bias_grad]
