import numpy as np

from mytorch.backends.cuda.env import CudaEnv, CudaMemory
from mytorch.backends.utils import calculate_reduce_shape
from mytorch.dtype import int64
from mytorch.backends.backend_dispatcher import BackendDispatcher


@BackendDispatcher.instance().register_backend_function("cuda", "max")
def max(tensor, dim, keepdim):
    from mytorch.tensor import Tensor, shape_size

    if dim is not None and not isinstance(dim, int):
        raise RuntimeError(f"max dimension must be a integer or None: {dim}")
    if dim is None:
        dim = tuple(range(len(tensor.shape)))
    else:
        dim = (dim,)

    output_shape = calculate_reduce_shape(tensor.shape, dim, keepdim)

    output_tensor = Tensor(
        dtype=tensor.dtype,
        shape=output_shape,
        device=tensor.device,
    )
    indices_tensor = Tensor(
        dtype=int64,
        shape=output_shape,
        device=tensor.device,
    )

    func_name = f"MaxReference_{tensor.dtype.name}"
    cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
    cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
        "max.cu", func_name, tensor.device.index
    )
    tensor_shape_num_bytes = len(tensor.shape) * np.dtype(np.int32).itemsize
    reduce_axis_num_bytes = len(dim) * np.dtype(np.int32).itemsize
    if tensor_shape_num_bytes + reduce_axis_num_bytes > 0:
        cuda_mem = CudaMemory(
            tensor.device.index, tensor_shape_num_bytes + reduce_axis_num_bytes
        )
        cuda_mem.write(np.array(list(tensor.shape) + list(dim), dtype=np.int32))
        tensor_shape_ptr = int(cuda_mem.ptr)
        reduce_axis_ptr = tensor_shape_ptr + tensor_shape_num_bytes
    else:
        tensor_shape_ptr = reduce_axis_ptr = 0
    grid_dim = (1, 32, 1)
    block_dim = (1024, 1, 1)
    num_shared_bytes = (block_dim[0] // 32) * (
        np.dtype(np.int64).itemsize + tensor.dtype.itemsize()
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
            np.array(len(dim), np.int32),
            np.array(reduce_axis_ptr, np.uint64),
            indices_tensor,
            output_tensor,
        ],
        num_shared_bytes=num_shared_bytes,
    )

    return output_tensor, indices_tensor


@BackendDispatcher.instance().register_backend_function("cuda", "max_backward")
def max_backward(output_grad, indices_tensor, tensor, dim, keepdim):
    from mytorch.tensor import Tensor, shape_size

    if dim is None:
        dim = tuple(range(len(tensor.shape)))
    else:
        dim = (dim,)

    input_grad = Tensor(dtype=tensor.dtype, shape=tensor.shape, device=tensor.device)
    func_name = f"max_backward_reference_{tensor.dtype.name}"
    cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
    cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
        "max.cu", func_name, tensor.device.index
    )
    tensor_shape_num_bytes = len(tensor.shape) * np.dtype(np.int32).itemsize
    reduce_axis_num_bytes = len(dim) * np.dtype(np.int32).itemsize
    if tensor_shape_num_bytes + reduce_axis_num_bytes > 0:
        cuda_mem = CudaMemory(
            tensor.device.index, tensor_shape_num_bytes + reduce_axis_num_bytes
        )
        cuda_mem.write(np.array(list(tensor.shape) + list(dim), dtype=np.int32))
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
            np.array(len(dim), np.int32),
            np.array(reduce_axis_ptr, np.uint64),
            indices_tensor,
            input_grad,
            output_grad,
        ],
    )

    return [input_grad]
