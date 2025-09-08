import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional

from mytorch.backends.backend_dispatcher import BackendDispatcher
from mytorch.tensor import Tensor


def _im2col(input: npt.NDArray, kernel_size: Tuple[int, int], stride: Tuple[int, int], padding: Tuple[int, int]):
    batch_size, channels, height, width = input.shape
    output_height = (height + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    output_width = (width + 2 * padding[1] - kernel_size[1]) // stride[1] + 1

    im = np.pad(input, [[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]]], 'constant')
    col = np.zeros((batch_size, channels, *kernel_size, output_height, output_width))

    for y in range(kernel_size[0]):
        y_max = y + stride[0] * output_height
        for x in range(kernel_size[1]):
            x_max = x + stride[1] * output_width
            col[:, :, y, x, :, :] = im[:, :, y: y_max: stride[0], x: x_max: stride[1]]

    col = col.transpose(0, 4, 5, 1, 2, 3).copy().reshape(batch_size * output_height * output_width, -1)
    return col


def _col2im(col: npt.NDArray, input_shape: Tuple[int, ...], kernel_size: Tuple[int, int], stride: Tuple[int, int],
            padding: Tuple[int, int]):
    batch_size, channels, height, width = input_shape
    output_height = (height + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    output_width = (width + 2 * padding[1] - kernel_size[1]) // stride[1] + 1

    col = col.reshape(batch_size, output_height, output_width, channels, *kernel_size).transpose(0, 3, 4, 5, 1, 2)

    im = np.zeros((batch_size, channels,
                   height + 2 * padding[0] + stride[0] - 1, width + 2 * padding[1] + stride[1] - 1))
    for y in range(kernel_size[0]):
        y_max = y + stride[0] * output_height
        for x in range(kernel_size[1]):
            x_max = x + stride[1] * output_width
            im[:, :, y: y_max: stride[0], x: x_max: stride[1]] += col[:, :, y, x, :, :]

    return im[:, :, padding[0]: height + padding[0], padding[1]: width + padding[1]]


@BackendDispatcher.instance().register_backend_function("cpu", "conv2d")
def cpu_conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor], stride: Tuple[int, int],
               padding: Tuple[int, int]):
    batch_size = input.shape[0]
    output_height = (input.shape[2] + 2 * padding[0] - weight.shape[2]) // stride[0] + 1
    output_width = (input.shape[3] + 2 * padding[1] - weight.shape[3]) // stride[1] + 1
    col = _im2col(input._numpy(), weight.shape[2:], stride, padding)
    weight_np_array = weight._numpy().reshape(weight.shape[0], -1).T
    output_np_array = np.matmul(col, weight_np_array)
    if bias is not None:
        output_np_array += bias._numpy()
    output_np_array = np.transpose(output_np_array.reshape(batch_size, output_height, output_width, -1), (0, 3, 1, 2))
    output_np_array = output_np_array.astype(input.dtype.np_dtype)
    return Tensor(cpu_array=output_np_array, dtype=input.dtype, device=input.device)


@BackendDispatcher.instance().register_backend_function("cpu", "conv2d_backward")
def cpu_conv2d_backward(output_grad: Tensor, input: Tensor, weight: Tensor, bias: Optional[Tensor],
                        stride: Tuple[int, int], padding: Tuple[int, int]):
    input_np_array = _im2col(input._numpy(), weight.shape[2:], stride, padding)
    # [bhw, C_in * k^2]
    weight_np_array = weight._numpy().reshape(weight.shape[0], -1)
    # [C_out, C_in * k^2]
    output_grad_np_array = np.transpose(output_grad._numpy(), (0, 2, 3, 1)).reshape((-1, output_grad.shape[1]))
    # [bhw, C_out]
    input_grad_np_array = np.matmul(output_grad_np_array, weight_np_array).astype(input.dtype.np_dtype)
    input_grad_np_array = _col2im(input_grad_np_array, input.shape, weight.shape[2:], stride, padding)
    input_grad = Tensor(input_grad_np_array, device=input.device)
    weight_grad_np_array = np.matmul(output_grad_np_array.T, input_np_array).astype(weight.dtype.np_dtype)
    weight_grad_np_array = weight_grad_np_array.reshape(weight.shape)
    weight_grad = Tensor(weight_grad_np_array, device=input.device)
    if bias is not None:
        bias_grad_np_array = output_grad_np_array.sum(axis=0)
        bias_grad = Tensor(bias_grad_np_array, device=input.device)
    else:
        bias_grad = None
    return [input_grad, weight_grad, bias_grad]
