import numpy as np

from mytorch.tensor import Tensor
from mytorch.backends.cuda.env import CudaEnv
from mytorch.backends.backend_dispatcher import BackendDispatcher


def _pool_op(name, x, kernel_size, stride, padding):
    kernel_size = (
        (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    )
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding

    output_shape = [
        x.shape[0],
        x.shape[1],
        *[
            (x.shape[2 + i] + padding[i] * 2 - kernel_size[i]) // stride[i] + 1
            for i in range(2)
        ],
    ]

    tensor = Tensor(
        shape=output_shape,
        dtype=x.dtype,
        device=x.device,
    )
    cuda_kernel_and_stream_manager = (
        CudaEnv.instance().kernel_and_stream_manager
    )
    func_name = f"{name}_reference_{x.dtype.name}"
    cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
        "pool_ops.cu", func_name, x.device.index
    )
    cuda_kernel.run(
        (1, 4, 4),
        (32, 4, 4),
        [
            np.array(x.shape[0]),
            np.array(x.shape[1]),
            np.array(x.shape[2]),
            np.array(x.shape[3]),
            np.array(kernel_size[0]),
            np.array(kernel_size[1]),
            np.array(stride[0]),
            np.array(stride[1]),
            np.array(padding[0]),
            np.array(padding[1]),
            x,
            tensor,
        ],
    )

    return tensor


def _pool_op_backward(name, output_grad, output, input, kernel_size, stride, padding):
    kernel_size = (
        (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    )
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding

    input_grad = Tensor(shape=input.shape, dtype=input.dtype, device=input.device)

    func_name = f"{name}_backward_reference_{input.dtype.name}"
    cuda_kernel_and_stream_manager = (
        CudaEnv.instance().kernel_and_stream_manager
    )
    cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
        "pool_ops.cu", func_name, input.device.index
    )
    input_grad.fill_(0)
    cuda_kernel.run(
        (1, 4, 4),
        (32, 4, 4),
        [
            np.array(input.shape[0]),
            np.array(input.shape[1]),
            np.array(input.shape[2]),
            np.array(input.shape[3]),
            np.array(kernel_size[0]),
            np.array(kernel_size[1]),
            np.array(stride[0]),
            np.array(stride[1]),
            np.array(padding[0]),
            np.array(padding[1]),
            input,
            output,
            input_grad,
            output_grad,
        ],
    )

    return [input_grad]


@BackendDispatcher.instance().register_backend_function("cuda", "max_pool2d")
def cuda_max_pool2d(x, kernel_size, stride, padding):
    return _pool_op("max_pool2d", x, kernel_size, stride, padding)


@BackendDispatcher.instance().register_backend_function("cuda", "max_pool2d_backward")
def cuda_max_pool2d_backward(output_grad, output, input, kernel_size, stride, padding):
    return _pool_op_backward(output_grad, output, input, kernel_size, stride, padding)
