import numpy as np

from mytorch.ops.reduce_ops import _calculate_reduce_shape
from mytorch.tensor import (
    Tensor,
    InvalidDataTypeError,
    InvalidDeviceError,
    CudaMemory,
    shape_size,
)
from mytorch.cuda.env import CudaEnv
from mytorch.autograd import DAGTracker


def max(tensor, reduce_axis=None, keepdim=False):
    assert reduce_axis is None or isinstance(reduce_axis, int)
    if reduce_axis is None:
        return_indices = False
        reduce_axis = tuple(range(len(tensor.shape)))
    else:
        return_indices = True
        reduce_axis = (reduce_axis,)
    output_shape = _calculate_reduce_shape(tensor.shape, reduce_axis, keepdim)

    requires_grad = tensor.requires_grad

    output_tensor = Tensor(
        dtype=tensor.dtype,
        shape=output_shape,
        device=tensor.device,
        requires_grad=requires_grad,
    )
    indices_tensor = Tensor(
        dtype=np.int64,
        shape=output_shape,
        device=tensor.device,
        requires_grad=False,
    )

    if tensor.device.type == "cuda":
        if tensor.dtype == np.float32:
            func_name = "max_reference_fp32"
        elif tensor.dtype == np.float16:
            func_name = "max_reference_fp16"
        else:
            raise InvalidDataTypeError(tensor.dtype)
        cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
        cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
            "max.cu", func_name, tensor.device.index
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
        grid_dim = (1, 32, 1)
        block_dim = (1024, 1, 1)
        num_shared_bytes = (block_dim[0] // 32) * (
            np.dtype(np.int64).itemsize + np.dtype(tensor).itemsize
        )
        num_elements = shape_size(tensor.shape)
        cuda_kernel.run(
            grid_dim,
            block_dim,
            [
                np.array(num_elements, np.int32),
                tensor,
                np.array(len(tensor.shape), np.int32),
                np.array(tensor_shape_ptr, np.uint64),
                np.array(len(reduce_axis), np.int32),
                np.array(reduce_axis_ptr, np.uint64),
                indices_tensor,
                output_tensor,
            ],
            num_shared_bytes=num_shared_bytes,
        )

    elif tensor.device.type == "cpu":
        raise NotImplementedError()

    else:
        raise InvalidDeviceError(tensor.device.type)

    if requires_grad:
        DAGTracker.instance().add_node(
            "max", [tensor, reduce_axis, keepdim], [output_tensor], [indices_tensor]
        )

    if return_indices:
        return output_tensor, indices_tensor
    else:
        return output_tensor


@DAGTracker.instance().register_backward_function("max")
def max_backward(output_grad, indices_tensor, tensor, reduce_axis, keepdim):
    input_grad = Tensor(dtype=tensor.dtype, shape=tensor.shape, device=tensor.device)

    if tensor.device.type == "cuda":
        if tensor.dtype == np.float32:
            func_name = "max_backward_reference_fp32"
        elif tensor.dtype == np.float16:
            func_name = "max_backward_reference_fp16"
        else:
            raise InvalidDataTypeError(tensor.dtype)
        cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
        cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
            "max.cu", func_name, tensor.device.index
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
        grid_dim = (1, 32, 1)
        block_dim = (1024, 1, 1)
        num_elements = shape_size(tensor.shape)
        cuda_kernel.run(
            grid_dim,
            block_dim,
            [
                np.array(num_elements, np.int32),
                tensor,
                np.array(len(tensor.shape), np.int32),
                np.array(tensor_shape_ptr, np.uint64),
                np.array(len(reduce_axis), np.int32),
                np.array(reduce_axis_ptr, np.uint64),
                indices_tensor,
                input_grad,
                output_grad,
            ],
        )

    elif tensor.device.type == "cpu":
        raise NotImplementedError()

    else:
        raise InvalidDeviceError(tensor.device.type)

    return [input_grad]
