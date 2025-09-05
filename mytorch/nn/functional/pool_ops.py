import numpy as np

from mytorch.tensor import Tensor, InvalidDeviceError
from mytorch.backends.cuda.env import CudaEnv
from mytorch.autograd import DAGTracker


def pool_operation_forward(name):
    def forward(x, kernel_size, stride=None, padding=0):
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

        requires_grad = x.requires_grad

        if x.device.type == "cuda":
            tensor = Tensor(
                shape=output_shape,
                dtype=x.dtype,
                device=x.device,
                requires_grad=requires_grad,
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

        elif x.device.type == "cpu":
            raise NotImplementedError()

        else:
            raise InvalidDeviceError(x.device.type)

        if requires_grad:
            DAGTracker.instance().add_node(
                name, [x, kernel_size, stride, padding], [tensor], [tensor]
            )

        return tensor

    return forward


def pool_operation_backward(name):
    @DAGTracker.instance().register_backward_function(name)
    def backward(output_grad, output, input, kernel_size, stride, padding):
        kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        stride = (stride, stride) if isinstance(stride, int) else stride
        padding = (padding, padding) if isinstance(padding, int) else padding

        output_shape = [
            input.shape[0],
            input.shape[1],
            *[
                (input.shape[2 + i] + padding[i] * 2 - kernel_size[i]) // stride[i] + 1
                for i in range(2)
            ],
        ]

        input_grad = Tensor(shape=input.shape, dtype=input.dtype, device=input.device)

        if input.device.type == "cuda":
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

        elif input.device.type == "cpu":
            raise NotImplementedError()

        else:
            raise InvalidDeviceError(input.device.type)

        return [input_grad]

    return backward


max_pool2d = pool_operation_forward("max_pool2d")
max_pool2d_backward = pool_operation_backward("max_pool2d")
