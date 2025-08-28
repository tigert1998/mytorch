import numpy as np

from mytorch.autograd import DAGTracker
from mytorch.tensor import Tensor, InvalidDeviceError, InvalidDataTypeError
from mytorch.cuda.env import CudaEnv
from mytorch.ops.basic_ops import _cuda_bmm


def _im2col_input(input, weight, bias, stride=1, padding=0):
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

    if input.device.type == "cuda":
        num_rols = weight.shape[1] * weight.shape[2] * weight.shape[3] + int(
            bias is not None
        )
        num_rols_padded = (num_rols + 3) // 4 * 4
        a_shape = [
            1,
            output_shape[0] * output_shape[2] * output_shape[3],
            num_rols_padded,
        ]
        a_tensor = Tensor(
            shape=a_shape,
            dtype=input.dtype,
            device=input.device,
        )
        cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
        assert input.dtype == weight.dtype and (
            bias is None or input.dtype == bias.dtype
        )
        if input.dtype == np.float32:
            func_name = "im2col_input_reference_fp32"
        elif input.dtype == np.float16:
            func_name = "im2col_input_reference_fp16"
        else:
            raise InvalidDataTypeError(input.dtype)
        cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
            "im2col.cu", func_name, input.device.index
        )
        block_dim = [32, 32, 1]
        grid_dim = [32, 32, 1]
        cuda_kernel.run(
            grid_dim,
            block_dim,
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
                bias,
                a_tensor,
            ],
        )

    elif input.device.type == "cpu":
        raise NotImplementedError()

    else:
        raise InvalidDeviceError(input.device.type)

    return a_tensor


def _reverse_im2col_input(a_tensor, input, weight, bias, stride=1, padding=0):
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

    input_grad = Tensor(shape=input.shape, device=input.device, dtype=input.dtype)

    if input.device.type == "cuda":
        cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
        assert input.dtype == weight.dtype and (
            bias is None or input.dtype == bias.dtype
        )
        if input.dtype == np.float32:
            func_name = "reverse_im2col_input_reference_fp32"
        elif input.dtype == np.float16:
            func_name = "reverse_im2col_input_reference_fp16"
        else:
            raise InvalidDataTypeError(input.dtype)
        cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
            "im2col.cu", func_name, input.device.index
        )
        block_dim = [8, 8, 8]
        shape = [output_shape[0], output_shape[2], output_shape[3]]
        grid_dim = [(i + j - 1) // j * j for i, j in zip(shape, block_dim)]
        input_grad.fill_(0)
        cuda_kernel.run(
            grid_dim,
            block_dim,
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
                input_grad,
                bias,
                a_tensor,
            ],
        )

    elif input.device.type == "cpu":
        raise NotImplementedError()

    else:
        raise InvalidDeviceError(input.device.type)

    return input_grad


def _im2col_weight(input, weight, bias, stride=1, padding=0):
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

    if input.device.type == "cuda":
        num_rols = weight.shape[1] * weight.shape[2] * weight.shape[3] + int(
            bias is not None
        )
        num_rols_padded = (num_rols + 3) // 4 * 4
        cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager

        b_shape = [1, output_shape[1], num_rols_padded]
        b_tensor = Tensor(
            shape=b_shape,
            dtype=input.dtype,
            device=input.device,
        )
        if input.dtype == np.float32:
            func_name = "im2col_weight_reference_fp32"
        elif input.dtype == np.float16:
            func_name = "im2col_weight_reference_fp16"
        else:
            raise InvalidDataTypeError(input.dtype)
        cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
            "im2col.cu", func_name, input.device.index
        )
        block_dim = [32, 32, 1]
        grid_dim = [32, 32, 1]
        cuda_kernel.run(
            grid_dim,
            block_dim,
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
                weight,
                bias,
                b_tensor,
            ],
        )

    elif input.device.type == "cpu":
        raise NotImplementedError()

    else:
        raise InvalidDeviceError(input.device.type)

    return b_tensor


def _reverse_im2col_weight(b_tensor, input, weight, bias, stride=1, padding=0):
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding

    weight_grad = Tensor(shape=weight.shape, dtype=weight.dtype, device=weight.device)
    if bias is not None:
        bias_grad = Tensor(shape=bias.shape, dtype=bias.dtype, device=bias.device)
    else:
        bias_grad = None

    if input.device.type == "cuda":
        cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
        if input.dtype == np.float32:
            func_name = "reverse_im2col_weight_reference_fp32"
        elif input.dtype == np.float16:
            func_name = "reverse_im2col_weight_reference_fp16"
        else:
            raise InvalidDataTypeError(input.dtype)
        cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
            "im2col.cu", func_name, input.device.index
        )
        block_dim = [32, 32, 1]
        grid_dim = [32, 32, 1]
        cuda_kernel.run(
            grid_dim,
            block_dim,
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
                weight_grad,
                bias_grad,
                b_tensor,
            ],
        )

    elif input.device.type == "cpu":
        raise NotImplementedError()

    else:
        raise InvalidDeviceError(input.device.type)

    return weight_grad, bias_grad


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

    requires_grad = (
        input.requires_grad
        or weight.requires_grad
        or (bias is not None and bias.requires_grad)
    )

    if input.device.type == "cuda":
        a_tensor = _im2col_input(input, weight, bias, stride, padding)
        b_tensor = _im2col_weight(input, weight, bias, stride, padding)
        c_tensor = _cuda_bmm(a_tensor, b_tensor, False, True, False)
        tensor = c_tensor.reshape(
            (output_shape[0], output_shape[2], output_shape[3], output_shape[1])
        ).permute((0, 3, 1, 2))
        tensor.requires_grad = requires_grad

    elif input.device.type == "cpu":
        raise NotImplementedError()

    else:
        raise InvalidDeviceError(input.device.type)

    if requires_grad:
        DAGTracker.instance().add_node(
            "conv2d", [input, weight, bias, stride, padding], [tensor]
        )

    return tensor


@DAGTracker.instance().register_backward_function("conv2d")
def conv2d_backward(output_grad, input, weight, bias=None, stride=1, padding=0):
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding

    if input.device.type == "cuda":
        assert (
            input.dtype == weight.dtype
            and (bias is None or input.dtype == bias.dtype)
            and input.dtype == output_grad.dtype
        )
        a_tensor = _im2col_input(input, weight, bias, stride, padding)
        # [1, bhw, padded(C_in * k^2)]
        b_tensor = _im2col_weight(input, weight, bias, stride, padding)
        # [1, C_out, padded(C_in * k^2)]
        c_tensor = output_grad.permute((0, 2, 3, 1)).reshape(
            (1, -1, output_grad.shape[1])
        )
        # [1, bhw, C_out]
        a_tensor_grad = _cuda_bmm(c_tensor, b_tensor, False, False, False)
        b_tensor_grad = _cuda_bmm(c_tensor, a_tensor, True, False, False)
        input_grad = _reverse_im2col_input(
            a_tensor_grad, input, weight, bias, stride, padding
        )
        weight_grad, bias_grad = _reverse_im2col_weight(
            b_tensor_grad, input, weight, bias, stride, padding
        )

    elif input.device.type == "cpu":
        raise NotImplementedError()

    else:
        raise InvalidDeviceError(input.device.type)

    return [input_grad, weight_grad, bias_grad]
