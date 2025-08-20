import numpy as np

from autograd import DAGTracker
from tensor import Tensor, InvalidDeviceError, InvalidDataTypeError
from cuda.cuda_utils import CudaKernelAndStreamManager


def conv2d(input, weight, bias=None, stride=1, padding=0):
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding

    output_shape = [
        input.shape[0],
        weight.shape[0],
        *[
            (input.shape[2 + i] + padding[i] * 2 - weight.shape[2 + i]) // stride[i] + 1
            for i in range(2)
        ],
    ]

    dag_tracker = DAGTracker.instance()

    if input.device.type == "cuda":
        tensor = Tensor(
            shape=output_shape,
            dtype=input.dtype,
            device=input.device,
            requires_grad=True,
        )
        cuda_kernel_and_stream_manager = CudaKernelAndStreamManager.instance()
        assert input.dtype == weight.dtype and (
            bias is None or input.dtype == bias.dtype
        )
        if input.dtype == np.float32:
            func_name = "conv2d_reference_fp32"
        elif input.dtype == np.float16:
            func_name = "conv2d_reference_fp16"
        else:
            raise InvalidDataTypeError(input.dtype)
        cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
            "conv2d.cu", func_name, input.device.index
        )
        cuda_kernel.run(
            (output_shape[0] * output_shape[1], output_shape[2], output_shape[3]),
            (1, 1, 1),
            [
                np.array(input.shape[0]),
                np.array(input.shape[2]),
                np.array(input.shape[3]),
                np.array(input.shape[1]),
                np.array(weight.shape[0]),
                np.array(weight.shape[2]),
                np.array(weight.shape[3]),
                np.array(stride[0]),
                np.array(stride[1]),
                np.array(padding[0]),
                np.array(padding[1]),
                input,
                weight,
                bias,
                tensor,
            ],
        )

    elif input.device.type == "cpu":
        ...

    else:
        raise InvalidDeviceError(input.device.type)

    dag_tracker.add_node("conv2d", [input, weight, bias, stride, padding], [tensor])

    return tensor


@DAGTracker.instance().register_backward_function("conv2d")
def conv2d_backward(output_grad, input, weight, bias=None, stride=1, padding=0):
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding

    output_shape = [
        input.shape[0],
        weight.shape[0],
        *[
            (input.shape[2 + i] + padding[i] * 2 - weight.shape[2 + i]) // stride[i] + 1
            for i in range(2)
        ],
    ]

    input_grad = Tensor(shape=input.shape, dtype=input.dtype, device=input.device)
    weight_grad = Tensor(shape=weight.shape, dtype=weight.dtype, device=weight.device)
    bias_grad = (
        None
        if bias is None
        else Tensor(shape=bias.shape, dtype=bias.dtype, device=bias.device)
    )

    if input.device.type == "cuda":
        assert (
            input.dtype == weight.dtype
            and (bias is None or input.dtype == bias.dtype)
            and input.dtype == output_grad.dtype
        )
        if input.dtype == np.float32:
            func_name = "conv2d_backward_reference_fp32"
        elif input.dtype == np.float16:
            func_name = "conv2d_backward_reference_fp16"
        else:
            raise InvalidDataTypeError(input.dtype)
        cuda_kernel_and_stream_manager = CudaKernelAndStreamManager.instance()
        cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
            "conv2d.cu", func_name, input.device.index
        )
        cuda_kernel.run(
            (output_shape[0] * output_shape[1], output_shape[2], output_shape[3]),
            (1, 1, 1),
            [
                np.array(input.shape[0]),
                np.array(input.shape[2]),
                np.array(input.shape[3]),
                np.array(input.shape[1]),
                np.array(weight.shape[0]),
                np.array(weight.shape[2]),
                np.array(weight.shape[3]),
                np.array(stride[0]),
                np.array(stride[1]),
                np.array(padding[0]),
                np.array(padding[1]),
                input,
                weight,
                bias,
                input_grad,
                weight_grad,
                bias_grad,
                output_grad,
            ],
        )

    elif input.device.type == "cpu":
        ...

    else:
        raise InvalidDeviceError(input.device.type)

    return [input_grad, weight_grad, bias_grad]
