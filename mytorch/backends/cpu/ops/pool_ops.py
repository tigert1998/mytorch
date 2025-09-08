import numpy as np

from mytorch.tensor import Tensor
from mytorch.backends.backend_dispatcher import BackendDispatcher


@BackendDispatcher.instance().register_backend_function("cpu", "max_pool2d")
def cpu_max_pool2d(
    input_tensor: Tensor,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
):
    input = input_tensor._numpy()
    batch_size, channels, height, width = input.shape
    padded_input = np.pad(
        input,
        ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
        mode="constant",
    )

    output_height = (padding[0] * 2 + height - kernel_size[0]) // stride[0] + 1
    output_width = (padding[1] * 2 + width - kernel_size[1]) // stride[1] + 1

    windows = np.lib.stride_tricks.as_strided(
        padded_input,
        shape=(
            batch_size,
            channels,
            output_height,
            output_width,
            kernel_size[0],
            kernel_size[1],
        ),
        strides=(
            padded_input.strides[0],
            padded_input.strides[1],
            stride[0] * padded_input.strides[2],
            stride[1] * padded_input.strides[3],
            padded_input.strides[2],
            padded_input.strides[3],
        ),
    )

    output = windows.max(axis=(4, 5))

    return Tensor(output, device=input_tensor.device)


@BackendDispatcher.instance().register_backend_function("cpu", "max_pool2d_backward")
def cpu_max_pool2d_backward(
    output_grad_tensor: Tensor,
    output_tensor: Tensor,
    input_tensor: Tensor,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
):
    input = input_tensor._numpy()
    output_grad = output_grad_tensor._numpy()
    output = output_tensor._numpy()

    padded_input = np.pad(
        input_tensor._numpy(),
        pad_width=((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
        mode="constant",
    )

    window_shape = (
        input.shape[0],
        input.shape[1],
        output_grad.shape[2],
        output_grad.shape[3],
        kernel_size[0],
        kernel_size[1],
    )

    strides = (
        padded_input.strides[0],
        padded_input.strides[1],
        stride[0] * padded_input.strides[2],
        stride[1] * padded_input.strides[3],
        padded_input.strides[2],
        padded_input.strides[3],
    )

    input_windows = np.lib.stride_tricks.as_strided(
        padded_input, shape=window_shape, strides=strides
    )

    output_expanded = output[..., np.newaxis, np.newaxis]
    max_mask = input_windows == output_expanded

    output_grad_expanded = output_grad[..., np.newaxis, np.newaxis]

    grad_windows = max_mask * output_grad_expanded

    grad_padded = np.zeros_like(padded_input)

    for i in range(output_grad.shape[2]):
        for j in range(output_grad.shape[3]):
            h_start = i * stride[0]
            h_end = h_start + kernel_size[0]
            w_start = j * stride[1]
            w_end = w_start + kernel_size[1]
            grad_padded[:, :, h_start:h_end, w_start:w_end] += grad_windows[
                :, :, i, j, :, :
            ]

    if padding[0] > 0:
        grad_padded = grad_padded[:, :, padding[0] : -padding[0], :]
    if padding[1] > 0:
        grad_padded = grad_padded[:, :, :, padding[1] : -padding[1]]

    return [Tensor(grad_padded, device=input_tensor.device)]
