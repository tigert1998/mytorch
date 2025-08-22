import numpy as np
from typing import Tuple, List

from mytorch.tensor import (
    InvalidDataTypeError,
    InvalidDeviceError,
    Tensor,
    shape_size,
    CudaMemory,
)
from mytorch.cuda.env import CudaEnv
from mytorch.autograd import DAGTracker


def _calculate_broadcast_shape(x_shape, y_shape):
    if len(x_shape) < len(y_shape):
        x_shape = (1,) * (len(y_shape) - len(x_shape)) + x_shape
    elif len(x_shape) > len(y_shape):
        y_shape = (1,) * (len(x_shape) - len(y_shape)) + y_shape
    ans = []
    for i, j in zip(x_shape, y_shape):
        assert i == j or i == 1 or j == 1
        ans.append(max(i, j))
    return tuple(ans)


def broadcast_binary_opeartion_forward(
    name, arg_types, no_grad_and_inplace, forward_op_cpu
):
    def forward(x, y, *args, **kwargs):
        from mytorch.elementwise_ops import extract_arg_list

        arg_list = extract_arg_list(arg_types, args, kwargs, x.dtype)

        shape = _calculate_broadcast_shape(x.shape, y.shape)

        requires_grad = not no_grad_and_inplace and (x.requires_grad or y.requires_grad)

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
                "broadcast_binary_ops.cu", func_name, x.device.index
            )

            x_shape_num_bytes = len(x.shape) * np.dtype(np.int32).itemsize
            y_shape_num_bytes = len(y.shape) * np.dtype(np.int32).itemsize
            if x_shape_num_bytes + y_shape_num_bytes > 0:
                cuda_mem = CudaMemory(x_shape_num_bytes + y_shape_num_bytes)
                cuda_mem.write(np.array(list(x.shape) + list(y.shape), dtype=np.int32))
                x_shape_ptr = int(cuda_mem.ptr)
                y_shape_ptr = x_shape_ptr + x_shape_num_bytes
            else:
                x_shape_ptr = y_shape_ptr = 0

            num_elements = shape_size(shape)

            if no_grad_and_inplace:
                output_tensor = np.array(0, np.uint64)
            else:
                output_tensor = Tensor(
                    dtype=x.dtype,
                    shape=shape,
                    device=x.device,
                    requires_grad=requires_grad,
                )

            cuda_kernel.run(
                ((num_elements + 255) // 256, 1, 1),
                (256, 1, 1),
                [
                    np.array(num_elements, dtype=np.int32),
                    np.array(len(x.shape), dtype=np.int32),
                    np.array(x_shape_ptr, dtype=np.uint64),
                    np.array(len(y.shape), dtype=np.int32),
                    np.array(y_shape_ptr, dtype=np.uint64),
                    x,
                    y,
                    *arg_list,
                    output_tensor,
                ],
            )

        elif x.device.type == "cpu":
            if no_grad_and_inplace:
                forward_op_cpu(x, y, *arg_list)
            else:
                output_tensor = Tensor(
                    dtype=x.dtype,
                    shape=shape,
                    device=x.device,
                    requires_grad=requires_grad,
                )
                output_tensor.cpu_array = forward_op_cpu(x, y, *arg_list)

        else:
            raise InvalidDeviceError(x.device.type)

        if requires_grad:
            DAGTracker.instance().add_node(name, [x, y, *arg_list], [output_tensor])

        if not no_grad_and_inplace:
            return output_tensor

    return forward


def broadcast_binary_opeartion_backward(name, backward_op_cpu):
    def tile_tensor(x, output_grad) -> Tuple[np.ndarray, List[int]]:
        x_shape = (1,) * (len(output_grad.shape) - len(x.shape)) + x.shape
        x_axis = [i for i in range(len(x_shape)) if x_shape[i] < output_grad.shape[i]]
        x_tile_reps = [
            output_grad.shape[i] if x_shape[i] < output_grad.shape[i] else 1
            for i in range(len(x_shape))
        ]
        x_tile = np.tile(x.cpu_array, x_tile_reps)
        return x_tile, x_axis

    @DAGTracker.instance().register_backward_function(name)
    def backward(output_grad, x, y, *args):
        x_grad = Tensor(dtype=x.dtype, shape=x.shape, device=x.device)
        x_grad.fill_(0)
        y_grad = Tensor(dtype=y.dtype, shape=y.shape, device=y.device)
        y_grad.fill_(0)

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
                "broadcast_binary_ops.cu", func_name, output_grad.device.index
            )

            x_shape_num_bytes = len(x.shape) * np.dtype(np.int32).itemsize
            y_shape_num_bytes = len(y.shape) * np.dtype(np.int32).itemsize
            if x_shape_num_bytes + y_shape_num_bytes > 0:
                cuda_mem = CudaMemory(x_shape_num_bytes + y_shape_num_bytes)
                cuda_mem.write(np.array(list(x.shape) + list(y.shape), dtype=np.int32))
                x_shape_ptr = int(cuda_mem.ptr)
                y_shape_ptr = x_shape_ptr + x_shape_num_bytes
            else:
                x_shape_ptr = y_shape_ptr = 0

            num_elements = shape_size(output_grad.shape)
            cuda_kernel.run(
                ((num_elements + 255) // 256, 1, 1),
                (256, 1, 1),
                [
                    np.array(num_elements, dtype=np.int32),
                    np.array(len(x.shape), dtype=np.int32),
                    np.array(x_shape_ptr, dtype=np.uint64),
                    np.array(len(y.shape), dtype=np.int32),
                    np.array(y_shape_ptr, dtype=np.uint64),
                    x,
                    y,
                    *args,
                    x_grad,
                    y_grad,
                    output_grad,
                ],
            )

        elif output_grad.device.type == "cpu":
            x_tile, x_axis = tile_tensor(x, output_grad)
            y_tile, y_axis = tile_tensor(y, output_grad)
            x_grad_cpu_array, y_grad_cpu_array = backward_op_cpu(
                x_tile, y_tile, *args, output_grad
            )
            x_grad.cpu_array = x_grad_cpu_array.sum(axis=tuple(x_axis)).reshape(x.shape)
            y_grad.cpu_array = y_grad_cpu_array.sum(axis=tuple(y_axis)).reshape(y.shape)
        else:
            raise InvalidDeviceError(output_grad.device.type)

        return [x_grad, y_grad]

    return backward


def _add_forward_op_cpu(x, y, alpha):
    return x.cpu_array + y.cpu_array * alpha


def _add_backward_op_cpu(x, y, alpha, output_grad):
    return output_grad.cpu_array, output_grad.cpu_array * alpha


add = broadcast_binary_opeartion_forward(
    "add", {"args": [(1, "default")], "kwargs": []}, False, _add_forward_op_cpu
)
add_backward = broadcast_binary_opeartion_backward("add", _add_backward_op_cpu)


def _sub_forward_op_cpu(x, y, alpha):
    return x.cpu_array - y.cpu_array * alpha


def _sub_backward_op_cpu(x, y, alpha, output_grad):
    return output_grad.cpu_array, -output_grad.cpu_array * alpha


sub = broadcast_binary_opeartion_forward(
    "sub", {"args": [(1, "default")], "kwargs": []}, False, _sub_forward_op_cpu
)
sub_backward = broadcast_binary_opeartion_backward("sub", _sub_backward_op_cpu)


def _mul_forward_op_cpu(x, y):
    return x.cpu_array * y.cpu_array


def _mul_backward_op_cpu(x, y, output_grad):
    return y * output_grad.cpu_array, x * output_grad.cpu_array


mul = broadcast_binary_opeartion_forward(
    "mul", {"args": [], "kwargs": []}, False, _mul_forward_op_cpu
)
mul_backward = broadcast_binary_opeartion_backward("mul", _mul_backward_op_cpu)


def _div_forward_op_cpu(x, y):
    return x.cpu_array / y.cpu_array


def _div_backward_op_cpu(x, y, output_grad):
    x_grad_cpu_array = 1 / y * output_grad.cpu_array
    y_grad_cpu_array = -x / (y**2) * output_grad.cpu_array
    return x_grad_cpu_array, y_grad_cpu_array


div = broadcast_binary_opeartion_forward(
    "div", {"args": [], "kwargs": []}, False, _div_forward_op_cpu
)
div_backward = broadcast_binary_opeartion_backward("div", _div_backward_op_cpu)


def _pow_forward_op_cpu(x, y):
    return np.power(x.cpu_array, y.cpu_array)


def _pow_backward_op_cpu(x, y, output_grad):
    x_grad_cpu_array = y * np.power(x, y - 1) * output_grad.cpu_array
    y_grad_cpu_array = np.power(x, y) * np.log(x) * output_grad.cpu_array
    return x_grad_cpu_array, y_grad_cpu_array


pow = broadcast_binary_opeartion_forward(
    "pow", {"args": [], "kwargs": []}, False, _pow_forward_op_cpu
)
pow_backward = broadcast_binary_opeartion_backward("pow", _pow_backward_op_cpu)


def _copy_forward_op_cpu(x, y):
    np.copyto(x.cpu_array, y.cpu_array)


_copy = broadcast_binary_opeartion_forward(
    "copy", {"args": [], "kwargs": []}, True, _copy_forward_op_cpu
)
