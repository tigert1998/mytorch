import numpy as np

from mytorch.tensor import (
    InvalidDataTypeError,
    InvalidDeviceError,
    CudaMemory,
    shape_size,
    Tensor,
)
from mytorch.cuda.env import CudaEnv
from mytorch.autograd import DAGTracker


def _calculate_reduce_shape(shape, axis, keepdim):
    assert np.all([0 <= i and i < len(shape) and isinstance(i, int) for i in axis])
    if keepdim:
        shape = [(1 if i in axis else shape_i) for i, shape_i in enumerate(shape)]
    else:
        shape = [shape_i for i, shape_i in enumerate(shape) if i not in axis]
    return tuple(shape)


def reduce_operation_forward(name, arg_types, forward_op_cpu):
    def forward(tensor, reduce_axis=None, keepdim=False, *args, **kwargs):
        from mytorch.elementwise_ops import extract_arg_list

        arg_list = extract_arg_list(arg_types, args, kwargs, tensor.dtype)

        if reduce_axis is None:
            reduce_axis = tuple(range(len(tensor.shape)))
        reduce_axis = tuple(sorted(reduce_axis))
        output_shape = _calculate_reduce_shape(tensor.shape, reduce_axis, keepdim)

        requires_grad = tensor.requires_grad

        output_tensor = Tensor(
            dtype=tensor.dtype,
            shape=output_shape,
            device=tensor.device,
            requires_grad=requires_grad,
        )
        output_tensor.fill_(0)

        if tensor.device.type == "cuda":
            if tensor.dtype == np.float32:
                func_name = f"{name}_reference_fp32"
            elif tensor.dtype == np.float16:
                func_name = f"{name}_reference_fp16"
            else:
                raise InvalidDataTypeError(tensor.dtype)
            cuda_kernel_and_stream_manager = (
                CudaEnv.instance().kernel_and_stream_manager
            )
            cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
                "reduce_ops.cu", func_name, tensor.device.index
            )

            tensor_shape_num_bytes = len(tensor.shape) * np.dtype(np.int32).itemsize
            reduce_axis_num_bytes = len(reduce_axis) * np.dtype(np.int32).itemsize
            if tensor_shape_num_bytes + reduce_axis_num_bytes > 0:
                cuda_mem = CudaMemory(tensor_shape_num_bytes + reduce_axis_num_bytes)
                cuda_mem.write(
                    np.array(list(tensor.shape) + list(reduce_axis), dtype=np.int32)
                )
                tensor_shape_ptr = int(cuda_mem.ptr)
                reduce_axis_ptr = tensor_shape_ptr + tensor_shape_num_bytes
            else:
                tensor_shape_ptr = reduce_axis_ptr = 0

            num_elements = shape_size(tensor.shape)
            cuda_kernel.run(
                (128, 1, 1),
                (128, 1, 1),
                [
                    np.array(num_elements, np.int32),
                    tensor,
                    *arg_list,
                    np.array(len(tensor.shape), np.int32),
                    np.array(tensor_shape_ptr, np.uint64),
                    np.array(len(reduce_axis), np.int32),
                    np.array(reduce_axis_ptr, np.uint64),
                    output_tensor,
                ],
            )

        elif tensor.device.type == "cpu":
            output_tensor.cpu_array = forward_op_cpu(
                tensor.cpu_array, reduce_axis, keepdim, *arg_list
            )

        else:
            raise InvalidDeviceError(tensor.device.type)

        if requires_grad:
            DAGTracker.instance().add_node(
                name, [tensor, reduce_axis, keepdim, *arg_list], [output_tensor]
            )

        return output_tensor

    return forward


def reduce_operation_backward(name, backward_op_cpu):
    @DAGTracker.instance().register_backward_function(name)
    def backward(output_grad, tensor, reduce_axis, keepdim, *args):
        from mytorch.tensor import Tensor

        tensor_grad = Tensor(
            shape=tensor.shape, dtype=tensor.dtype, device=tensor.device
        )

        if tensor.device.type == "cuda":
            if tensor.dtype == np.float32:
                func_name = f"{name}_backward_reference_fp32"
            elif tensor.dtype == np.float16:
                func_name = f"{name}_backward_reference_fp16"
            else:
                raise InvalidDataTypeError(tensor.dtype)
            cuda_kernel_and_stream_manager = (
                CudaEnv.instance().kernel_and_stream_manager
            )
            cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
                "reduce_ops.cu", func_name, tensor.device.index
            )

            tensor_shape_num_bytes = len(tensor.shape) * np.dtype(np.int32).itemsize
            reduce_axis_num_bytes = len(reduce_axis) * np.dtype(np.int32).itemsize
            if tensor_shape_num_bytes + reduce_axis_num_bytes > 0:
                cuda_mem = CudaMemory(tensor_shape_num_bytes + reduce_axis_num_bytes)
                cuda_mem.write(
                    np.array(list(tensor.shape) + list(reduce_axis), dtype=np.int32)
                )
                tensor_shape_ptr = int(cuda_mem.ptr)
                reduce_axis_ptr = tensor_shape_ptr + tensor_shape_num_bytes
            else:
                tensor_shape_ptr = reduce_axis_ptr = 0

            num_elements = shape_size(tensor.shape)
            cuda_kernel.run(
                (128, 1, 1),
                (128, 1, 1),
                [
                    np.array(num_elements),
                    tensor,
                    *args,
                    np.array(len(tensor.shape), np.int32),
                    np.array(tensor_shape_ptr, np.uint64),
                    np.array(len(reduce_axis), np.int32),
                    np.array(reduce_axis_ptr, np.uint64),
                    tensor_grad,
                    output_grad,
                ],
            )

        elif tensor.device.type == "cpu":
            np.copyto(
                tensor_grad.cpu_array,
                backward_op_cpu(output_grad.cpu_array, reduce_axis, keepdim, *args),
            )

        else:
            raise InvalidDeviceError(tensor.device.type)

        return [tensor_grad]

    return backward


def _sum_scale_forward_op_cpu(tensor_cpu_array, reduce_axis, keepdim, scale):
    return np.sum(tensor_cpu_array, axis=reduce_axis, keepdims=keepdim) * scale


def _sum_scale_backward_op_cpu(output_grad_cpu_array, reduce_axis, keepdim, scale):
    if keepdim:
        return output_grad_cpu_array * scale
    else:
        return np.expand_dims(output_grad_cpu_array, reduce_axis) * scale


_sum_scale = reduce_operation_forward(
    "sum_scale", {"args": [(1, "default")], "kwargs": []}, _sum_scale_forward_op_cpu
)

_sum_scale_backward = reduce_operation_backward("sum_scale", _sum_scale_backward_op_cpu)


def sum(x, dim=None, keepdim=False):
    return _sum_scale(x, dim, keepdim, 1)


def mean(x, dim=None, keepdim=False):
    if dim is None:
        dim = tuple(range(len(x.shape)))
    scale = 1 / shape_size([x.shape[i] for i in dim])
    return _sum_scale(x, dim, keepdim, scale)


def var(x, dim=None, correction=1, keepdim=False):
    if dim is None:
        dim = tuple(range(len(x.shape)))
    scale = 1 / (shape_size([x.shape[i] for i in dim]) - correction)
    return _sum_scale((x - mean(x, dim=dim, keepdim=True)) ** 2, dim, keepdim, scale)


def std(x, dim=None, correction=1, keepdim=False):
    return var(x, dim, correction, keepdim) ** 0.5
