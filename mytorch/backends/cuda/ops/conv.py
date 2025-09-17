from typing import Optional
import numpy as np

from mytorch.backends.cuda.env import CudaEnv
from mytorch.backends.backend_dispatcher import BackendDispatcher
from mytorch.backends.cuda.ops.mm import _cuda_bmm
from mytorch.autograd import no_grad
from mytorch.tensor import Tensor, MismatchDataTypesError


def _im2col_input(
    input: Tensor, weight: Tensor, bias: Optional[Tensor], stride=1, padding=0
):
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
    if not (
        input.dtype == weight.dtype and (bias is None or input.dtype == bias.dtype)
    ):
        dtypes = [input.dtype, weight.dtype]
        dtypes += [bias.dtype] if bias is not None else []
        raise MismatchDataTypesError(dtypes)
    func_name = f"im2col_input_reference_{input.dtype.name}"
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

    return a_tensor


def _col2im_input(
    a_tensor: Tensor,
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride=1,
    padding=0,
):
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding

    input_grad = Tensor(shape=input.shape, device=input.device, dtype=input.dtype)

    cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
    if not (
        input.dtype == weight.dtype and (bias is None or input.dtype == bias.dtype)
    ):
        dtypes = [input.dtype, weight.dtype]
        dtypes += [bias.dtype] if bias is not None else []
        raise MismatchDataTypesError(dtypes)
    func_name = f"col2im_input_reference_{input.dtype.name}"
    cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
        "im2col.cu", func_name, input.device.index
    )
    block_dim = [32, 32, 1]
    grid_dim = [32, 32, 1]
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

    return input_grad


def _im2col_weight(
    input: Tensor, weight: Tensor, bias: Optional[Tensor], stride=1, padding=0
):
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
    func_name = f"im2col_weight_reference_{input.dtype.name}"
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

    return b_tensor


def _col2im_weight(
    b_tensor: Tensor,
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride=1,
    padding=0,
):
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding

    weight_grad = Tensor(shape=weight.shape, dtype=weight.dtype, device=weight.device)
    if bias is not None:
        bias_grad = Tensor(shape=bias.shape, dtype=bias.dtype, device=bias.device)
    else:
        bias_grad = None

    cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
    func_name = f"col2im_weight_reference_{input.dtype.name}"
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

    return weight_grad, bias_grad


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

    a_tensor = _im2col_input(input, weight, bias, stride, padding)
    b_tensor = _im2col_weight(input, weight, bias, stride, padding)
    c_tensor = _cuda_bmm(a_tensor, b_tensor, False, True)
    tensor = c_tensor.reshape(
        (output_shape[0], output_shape[2], output_shape[3], output_shape[1])
    ).permute((0, 3, 1, 2))

    return tensor


@BackendDispatcher.instance().register_backend_function("cuda", "conv2d_backward")
def cuda_conv2d_backward(
    output_grad: Tensor,
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride,
    padding,
):
    a_tensor = _im2col_input(input, weight, bias, stride, padding)
    # [1, bhw, padded(C_in * k^2)]
    b_tensor = _im2col_weight(input, weight, bias, stride, padding)
    # [1, C_out, padded(C_in * k^2)]
    c_tensor = output_grad.permute((0, 2, 3, 1)).reshape((1, -1, output_grad.shape[1]))
    # [1, bhw, C_out]
    a_tensor_grad = _cuda_bmm(c_tensor, b_tensor, False, False)
    b_tensor_grad = _cuda_bmm(c_tensor, a_tensor, True, False)
    input_grad = _col2im_input(a_tensor_grad, input, weight, bias, stride, padding)
    weight_grad, bias_grad = _col2im_weight(
        b_tensor_grad, input, weight, bias, stride, padding
    )

    return [input_grad, weight_grad, bias_grad]
