import numpy as np
import numba

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
    pad_height = padding[0] * 2 + height
    pad_width = padding[1] * 2 + width
    padded_input = np.pad(
        input,
        ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
        mode="constant",
    )

    out_height = (pad_height - kernel_size[0]) // stride[0] + 1
    out_width = (pad_width - kernel_size[1]) // stride[1] + 1

    output = np.zeros((batch_size, channels, out_height, out_width)).astype(
        input_tensor.dtype.np_dtype
    )

    @numba.njit
    def loop(output, padded_input):
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * stride[0]
                        h_end = h_start + kernel_size[0]
                        w_start = w * stride[1]
                        w_end = w_start + kernel_size[1]

                        output[b, c, h, w] = np.max(
                            padded_input[b, c, h_start:h_end, w_start:w_end]
                        )

    loop(output, padded_input)

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
    batch_size, in_channels, input_height, input_width = input.shape
    _, _, output_height, output_width = output_tensor.shape
    output = output_tensor._numpy()
    output_grad = output_grad_tensor._numpy()
    kernel_height, kernel_width = kernel_size
    stride_height, stride_width = stride
    pad_height, pad_width = padding

    input_grad = np.zeros_like(input).astype(input_tensor.dtype.np_dtype)

    @numba.njit
    def loop(output_grad, input_grad, output, input):
        for b in range(batch_size):
            for c in range(in_channels):
                for oh in range(output_height):
                    for ow in range(output_width):
                        ih_start = oh * stride_height - pad_height
                        iw_start = ow * stride_width - pad_width
                        ih_end = min(
                            ih_start + kernel_height, input_height + pad_height
                        )
                        iw_end = min(iw_start + kernel_width, input_width + pad_width)

                        ih_start = max(0, ih_start)
                        iw_start = max(0, iw_start)

                        pool_region = input[b, c, ih_start:ih_end, iw_start:iw_end]

                        max_val = output[b, c, oh, ow]
                        mask = pool_region == max_val

                        input_grad[b, c, ih_start:ih_end, iw_start:iw_end] += (
                            mask * output_grad[b, c, oh, ow]
                        )

    loop(output_grad, input_grad, output, input)

    return [Tensor(input_grad, device=input_tensor.device)]
