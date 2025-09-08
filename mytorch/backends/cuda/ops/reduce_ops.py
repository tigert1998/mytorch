import numpy as np

from mytorch.backends.cuda.env import CudaEnv, CudaMemory
from mytorch.backends.utils import calculate_reduce_shape
from mytorch.backends.backend_dispatcher import BackendDispatcher


def _reduce_op(name, arg_list, tensor, dim, keepdim):
    from mytorch.tensor import shape_size, Tensor

    if dim is None:
        dim = tuple(range(len(tensor.shape)))
    dim = tuple(sorted(dim))
    output_shape = calculate_reduce_shape(tensor.shape, dim, keepdim)
    requires_grad = tensor.requires_grad
    output_tensor = Tensor(
        dtype=tensor.dtype,
        shape=output_shape,
        device=tensor.device,
        requires_grad=requires_grad,
    )
    func_name = f"{name}_reference_{tensor.dtype.name}"
    cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
    cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
        "reduce_ops.cu", func_name, tensor.device.index
    )
    tensor_shape_num_bytes = len(tensor.shape) * np.dtype(np.int32).itemsize
    dim_num_bytes = len(dim) * np.dtype(np.int32).itemsize
    if tensor_shape_num_bytes + dim_num_bytes > 0:
        cuda_mem = CudaMemory(
            tensor.device.index, tensor_shape_num_bytes + dim_num_bytes
        )
        cuda_mem.write(np.array(list(tensor.shape) + list(dim), dtype=np.int32))
        tensor_shape_ptr = int(cuda_mem.ptr)
        dim_ptr = tensor_shape_ptr + tensor_shape_num_bytes
    else:
        tensor_shape_ptr = dim_ptr = 0
    num_elements = shape_size(tensor.shape)
    block_dim = (1024, 1, 1)
    num_shared_bytes = (block_dim[0] // 32) * (
        np.dtype(np.int64).itemsize + tensor.dtype.itemsize()
    )
    cuda_kernel.run(
        (1, 32, 1),
        block_dim,
        [
            np.array(num_elements, np.int32),
            tensor,
            *arg_list,
            np.array(len(tensor.shape), np.int32),
            np.array(tensor_shape_ptr, np.uint64),
            np.array(len(dim), np.int32),
            np.array(dim_ptr, np.uint64),
            output_tensor,
        ],
        num_shared_bytes,
    )

    return output_tensor


def _reduce_op_backward(name, output_grad, tensor, dim, keepdim, args):
    from mytorch.tensor import Tensor, shape_size

    tensor_grad = Tensor(shape=tensor.shape, dtype=tensor.dtype, device=tensor.device)
    func_name = f"{name}_backward_reference_{tensor.dtype.name}"
    cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
    cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
        "reduce_ops.cu", func_name, tensor.device.index
    )
    tensor_shape_num_bytes = len(tensor.shape) * np.dtype(np.int32).itemsize
    dim_num_bytes = len(dim) * np.dtype(np.int32).itemsize
    if tensor_shape_num_bytes + dim_num_bytes > 0:
        cuda_mem = CudaMemory(
            tensor.device.index, tensor_shape_num_bytes + dim_num_bytes
        )
        cuda_mem.write(np.array(list(tensor.shape) + list(dim), dtype=np.int32))
        tensor_shape_ptr = int(cuda_mem.ptr)
        dim_ptr = tensor_shape_ptr + tensor_shape_num_bytes
    else:
        tensor_shape_ptr = dim_ptr = 0
    num_elements = shape_size(tensor.shape)
    cuda_kernel.run(
        (1, 32, 1),
        (1024, 1, 1),
        [
            np.array(num_elements),
            tensor,
            *args,
            np.array(len(tensor.shape), np.int32),
            np.array(tensor_shape_ptr, np.uint64),
            np.array(len(dim), np.int32),
            np.array(dim_ptr, np.uint64),
            tensor_grad,
            output_grad,
        ],
    )

    return [tensor_grad]


@BackendDispatcher.instance().register_backend_function("cuda", "sum_scale")
def cuda_sum_scale(x, dim, keepdim, scale):
    return _reduce_op("sum_scale", [np.array(scale, x.dtype.np_dtype)], x, dim, keepdim)


@BackendDispatcher.instance().register_backend_function("cuda", "sum_scale_backward")
def cuda_sum_scale_backward(output_grad, x, dim, keepdim, scale):
    return _reduce_op_backward(
        "sum_scale", output_grad, x, dim, keepdim, [np.array(scale, x.dtype.np_dtype)]
    )
