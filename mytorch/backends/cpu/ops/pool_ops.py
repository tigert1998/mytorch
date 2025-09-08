import numpy as np
import numba

from mytorch.tensor import Tensor
from mytorch.backends.backend_dispatcher import BackendDispatcher


@numba.njit(cache=True)
def _forward_loop(
    output,
    padded_input,
    stride,
    kernel_size,
):
    for b in range(output.shape[0]):
        for c in range(output.shape[1]):
            for h in range(output.shape[2]):
                for w in range(output.shape[3]):
                    h_start = h * stride[0]
                    h_end = h_start + kernel_size[0]
                    w_start = w * stride[1]
                    w_end = w_start + kernel_size[1]
                    output[b, c, h, w] = np.max(
                        padded_input[b, c, h_start:h_end, w_start:w_end]
                    )


@numba.njit(cache=True)
def _backward_loop(
    output_grad, input_grad, output, input, kernel_size, stride, padding
):
    for b in range(input_grad.shape[0]):
        for c in range(input_grad.shape[1]):
            for oh in range(output_grad.shape[2]):
                for ow in range(output_grad.shape[3]):
                    ih_start = oh * stride[0] - padding[0]
                    iw_start = ow * stride[1] - padding[1]
                    ih_end = min(
                        ih_start + kernel_size[0], input_grad.shape[2] + padding[0]
                    )
                    iw_end = min(
                        iw_start + kernel_size[1], input_grad.shape[3] + padding[1]
                    )
                    ih_start = max(0, ih_start)
                    iw_start = max(0, iw_start)
                    pool_region = input[b, c, ih_start:ih_end, iw_start:iw_end]
                    max_val = output[b, c, oh, ow]
                    mask = pool_region == max_val
                    input_grad[b, c, ih_start:ih_end, iw_start:iw_end] += (
                        mask * output_grad[b, c, oh, ow]
                    )


@BackendDispatcher.instance().register_backend_function("cpu", "max_pool2d")
def cpu_max_pool2d(
    input_tensor: Tensor,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
):
    input = input_tensor._numpy()
    batch_size, channels, height, width = input.shape
    pad_height = padding[0] * 2 + height
    pad_width = padding[1] * 2 + width
    padded_input = np.pad(
        input,
        ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
        mode="constant",
    )

    output_height = (pad_height - kernel_size[0]) // stride[0] + 1
    output_width = (pad_width - kernel_size[1]) // stride[1] + 1

    output = np.zeros((batch_size, channels, output_height, output_width)).astype(
        input_tensor.dtype.np_dtype
    )

    _forward_loop(
        output,
        padded_input,
        stride,
        kernel_size,
    )

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
    output = output_tensor._numpy()
    output_grad = output_grad_tensor._numpy()

    input_grad = np.zeros_like(input).astype(input_tensor.dtype.np_dtype)

    _backward_loop(output_grad, input_grad, output, input, kernel_size, stride, padding)

    return [Tensor(input_grad, device=input_tensor.device)]
