from typing import Tuple

import numpy as np

from mytorch.backends.cuda.env import CudaEnv
from mytorch.backends.backend_dispatcher import BackendDispatcher


@BackendDispatcher.instance().register_backend_function("cuda", "permute")
def cuda_permute(x, dims: Tuple[int, ...]):
    from mytorch.tensor import Tensor, CudaMemory, shape_size

    func_name = f"permute_reference_{x.dtype.name}"
    cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
    cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
        "basic_ops.cu",
        func_name,
        x.device.index,
    )
    output_tensor = Tensor(
        dtype=x.dtype,
        device=x.device,
        shape=tuple([x.shape[i] for i in dims]),
    )
    num_bytes = np.dtype(np.int32).itemsize * len(dims)
    if num_bytes > 0:
        cuda_mem = CudaMemory(num_bytes * 2)
        cuda_mem.write(np.array(x.shape + dims, dtype=np.int32))
        shape_ptr = int(cuda_mem.ptr)
        dims_ptr = shape_ptr + num_bytes
    else:
        shape_ptr = dims_ptr = 0
    num_elements = shape_size(x.shape)
    cuda_kernel.run(
        ((num_elements + 255) // 256, 1, 1),
        (256, 1, 1),
        [
            np.array(num_elements, np.int32),
            x,
            output_tensor,
            np.array(len(dims), np.int32),
            np.array(shape_ptr, dtype=np.uint64),
            np.array(dims_ptr, dtype=np.uint64),
        ],
    )
    return output_tensor


@BackendDispatcher.instance().register_backend_function("cuda", "permute_backward")
def cuda_permute_backward(output_grad, x, dims: Tuple[int, ...]):
    from mytorch.tensor import Tensor, CudaMemory, shape_size

    func_name = f"permute_backward_reference_{x.dtype.name}"
    cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
    cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
        "basic_ops.cu", func_name, x.device.index
    )
    input_grad = Tensor(
        dtype=x.dtype,
        device=x.device,
        shape=x.shape,
    )
    num_bytes = np.dtype(np.int32).itemsize * len(dims)
    if num_bytes > 0:
        cuda_mem = CudaMemory(num_bytes * 2)
        cuda_mem.write(np.array(x.shape + dims, dtype=np.int32))
        shape_ptr = int(cuda_mem.ptr)
        dims_ptr = shape_ptr + num_bytes
    else:
        shape_ptr = dims_ptr = 0
    num_elements = shape_size(x.shape)
    cuda_kernel.run(
        ((num_elements + 255) // 256, 1, 1),
        (256, 1, 1),
        [
            np.array(num_elements, dtype=np.int32),
            np.array(len(dims), dtype=np.int32),
            np.array(shape_ptr, dtype=np.uint64),
            np.array(dims_ptr, dtype=np.uint64),
            input_grad,
            output_grad,
        ],
    )
    return [input_grad]
