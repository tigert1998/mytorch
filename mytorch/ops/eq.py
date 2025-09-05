import numpy as np

from mytorch.tensor import (
    CudaMemory,
    Tensor,
    shape_size,
    InvalidDeviceError,
    MismatchDataTypesError,
)
from mytorch.backends.cuda.env import CudaEnv
from mytorch.dtype import int8


def eq(x: Tensor, y: Tensor):
    if x.dtype != y.dtype:
        raise MismatchDataTypesError([x.dtype, y.dtype])

    if x.device.type == "cuda":
        cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
        func_name = f"eq_reference_{x.dtype.name}"
        cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
            "eq.cu", func_name, x.device.index
        )
        output_tensor = Tensor(
            dtype=int8,
            shape=x.shape,
            device=x.device,
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

        num_elements = shape_size(x.shape)
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
                output_tensor,
            ],
        )
    elif x.device.type == "cpu":
        output_tensor = Tensor(
            dtype=int8,
            shape=x.shape,
            device=x.device,
        )
        output_tensor.cpu_array = (x._numpy() == y._numpy()).astype(np.int8)
    else:
        raise InvalidDeviceError(x.device.type)

    return output_tensor
