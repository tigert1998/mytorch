from functools import cache

import numpy as np

from mytorch.ops.cast import dtype_to_cpp_type, dtype_to_str
from mytorch.tensor import (
    CudaMemory,
    Tensor,
    shape_size,
    InvalidDeviceError,
    MismatchDataTypesError,
)
from mytorch.cuda.env import CudaEnv


@cache
def _generate_eq_cu():
    source = """#include <cuda_fp16.h>

#include <cuda/std/cstdint>
#include "broadcast_utils.cuh"

template <typename T>
__global__ void eq_reference(int n, int x_shape_n, int* x_shape, int y_shape_n,
                             int* y_shape, T* x, T* y, int8_t* output) {
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  if (xid >= n) return;
  int2 pair = broadcast(xid, x_shape_n, x_shape, y_shape_n, y_shape);
  output[xid] = (int8_t)(x[pair.x] == y[pair.y]);
}
"""
    dtypes = [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]
    for dtype in dtypes:
        source += f"""extern "C" __global__ void eq_reference_{dtype_to_str(dtype)}(int n, int x_shape_n, int* x_shape,
                                             int y_shape_n, int* y_shape,
                                             {dtype_to_cpp_type(dtype)}* x, {dtype_to_cpp_type(dtype)}* y, int8_t* output) {{
  eq_reference(n, x_shape_n, x_shape, y_shape_n, y_shape, x, y, output);
}}
"""
    return source


def eq(x, y):
    if x.dtype != y.dtype:
        raise MismatchDataTypesError([x.dtype, y.dtype])

    if x.device.type == "cuda":
        cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
        func_name = f"eq_reference_{dtype_to_str(x.dtype)}"
        cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
            "eq.cu", func_name, x.device.index, _generate_eq_cu()
        )
        output_tensor = Tensor(
            dtype=np.int8,
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
            dtype=np.int8,
            shape=x.shape,
            device=x.device,
        )
        output_tensor.cpu_array = (x == y).astype(np.int8)
    else:
        raise InvalidDeviceError(x.device.type)

    return output_tensor
