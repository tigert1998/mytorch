import numpy as np

from mytorch.tensor import InvalidDataTypeError, InvalidDeviceError, Tensor, shape_size
from mytorch.cuda.env import CudaEnv
from mytorch.autograd import DAGTracker


def extract_arg_list(arg_types, args, kwargs, default_dtype):
    def cast_dtype(dtype):
        if dtype == "default":
            return default_dtype
        return dtype

    return [
        np.array(args[i] if i < len(args) else default, dtype=cast_dtype(dtype))
        for i, (default, dtype) in enumerate(arg_types["args"])
    ] + [
        np.array(kwargs.get(name, default), dtype=cast_dtype(dtype))
        for name, default, dtype in arg_types["kwargs"]
    ]


def elementwise_operation_forward(name, arg_types, no_grad_and_inplace, forward_op_cpu):

    def forward(x, *args, **kwargs):
        arg_list = extract_arg_list(arg_types, args, kwargs, x.dtype)
        if x.device.type == "cuda":
            if x.dtype == np.float32:
                func_name = f"{name}_reference_fp32"
            elif x.dtype == np.float16:
                func_name = f"{name}_reference_fp16"
            else:
                raise InvalidDataTypeError(x.dtype)
            cuda_kernel_and_stream_manager = (
                CudaEnv.instance().kernel_and_stream_manager
            )
            cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
                "elementwise_ops.cu", func_name, x.device.index
            )

            if no_grad_and_inplace:
                output_tensor = np.array(0, np.uint64)
            else:
                output_tensor = Tensor(
                    dtype=x.dtype, shape=x.shape, device=x.device, requires_grad=True
                )

            num_elements = shape_size(x.shape)
            cuda_kernel.run(
                ((num_elements + 255) // 256, 1, 1),
                (256, 1, 1),
                [
                    np.array(num_elements, dtype=np.int32),
                    x,
                    *arg_list,
                    output_tensor,
                ],
            )

        elif x.device.type == "cpu":
            output_tensor = Tensor(
                dtype=x.dtype,
                shape=x.shape,
                device=x.device,
                requires_grad=not no_grad_and_inplace,
            )
            output_tensor.cpu_array = forward_op_cpu(x, *arg_list)

        else:
            raise InvalidDeviceError(x.device.type)

        if not no_grad_and_inplace:
            DAGTracker.instance().add_node(name, [x, *arg_list], [output_tensor])

        return output_tensor

    return forward


def elementwise_operation_backward(name, backward_op_cpu):
    @DAGTracker.instance().register_backward_function(name)
    def backward(output_grad, x, *args):
        x_grad = Tensor(dtype=x.dtype, shape=x.shape, device=x.device)
        x_grad.fill_(0)

        if output_grad.device.type == "cuda":
            if output_grad.dtype == np.float32:
                func_name = f"{name}_backward_reference_fp32"
            elif output_grad.dtype == np.float16:
                func_name = f"{name}_backward_reference_fp16"
            else:
                raise InvalidDataTypeError(output_grad.dtype)
            cuda_kernel_and_stream_manager = (
                CudaEnv.instance().kernel_and_stream_manager
            )
            cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
                "elementwise_ops.cu", func_name, output_grad.device.index
            )

            num_elements = shape_size(output_grad.shape)
            cuda_kernel.run(
                ((num_elements + 255) // 256, 1, 1),
                (256, 1, 1),
                [
                    np.array(num_elements, dtype=np.int32),
                    x,
                    *args,
                    x_grad,
                    output_grad,
                ],
            )

        elif output_grad.device.type == "cpu":
            x_grad.cpu_array = backward_op_cpu(x, *args, output_grad)
        else:
            raise InvalidDeviceError(output_grad.device.type)

        return [x_grad]

    return backward


def _fill_forward_op_cpu(x, value):
    np.copyto(x.cpu_array, value)


_fill = elementwise_operation_forward(
    "fill", {"args": [(1, "default")], "kwargs": []}, True, _fill_forward_op_cpu
)
