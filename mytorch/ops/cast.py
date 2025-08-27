from functools import cache

import numpy as np

from mytorch.tensor import InvalidDataTypeError, Tensor, shape_size, InvalidDeviceError
from mytorch.cuda.env import CudaEnv
from mytorch.autograd import DAGTracker


def dtype_to_str(dtype):
    if dtype == np.int8:
        return "int8"
    elif dtype == np.int16:
        return "int16"
    elif dtype == np.int32:
        return "int32"
    elif dtype == np.int64:
        return "int64"
    elif dtype == np.float16:
        return "fp16"
    elif dtype == np.float32:
        return "fp32"
    elif dtype == np.float64:
        return "fp64"
    else:
        raise InvalidDataTypeError(dtype)


def dtype_to_cpp_type(dtype):
    if dtype == np.int8:
        return "int8_t"
    elif dtype == np.int16:
        return "int16_t"
    elif dtype == np.int32:
        return "int32_t"
    elif dtype == np.int64:
        return "int64_t"
    elif dtype == np.float16:
        return "half"
    elif dtype == np.float32:
        return "float"
    elif dtype == np.float64:
        return "double"
    else:
        raise InvalidDataTypeError(dtype)


@cache
def _generate_cast_cu():
    source = """#include <cuda_fp16.h>
#include <cuda/std/cstdint>

template <typename T1, typename T2>
__global__ void cast_reference(int n, T1* input, T2* output) {
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize + blockIdx.x * blockDim.x / warpSize;
  int num_warps = gridDim.x * blockDim.x / warpSize;
  for (int i = warp_id * warpSize; i < n; i += num_warps * warpSize) {
    int xid = i + lane_id;
    if (xid < n) output[xid] = (T2)input[xid];
  }
}
"""

    dtypes = [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]
    for s_dtype in dtypes:
        for t_dtype in dtypes:
            if s_dtype != t_dtype:
                source += f"""
extern "C" __global__ void cast_reference_{dtype_to_str(s_dtype)}_{dtype_to_str(t_dtype)}(int n, {dtype_to_cpp_type(s_dtype)}* input, {dtype_to_cpp_type(t_dtype)}* output)  {{
    cast_reference(n, input, output);
}}
"""

    return source


def _cast(x, dtype):
    if x.dtype == dtype:
        return x

    requires_grad = (
        x.requires_grad
        and np.issubdtype(dtype, np.floating)
        and np.issubdtype(x.dtype, np.floating)
    )

    if x.device.type == "cuda":
        cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
        func_name = f"cast_reference_{dtype_to_str(x.dtype)}_{dtype_to_str(dtype)}"
        cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
            "cast.cu", func_name, x.device.index, _generate_cast_cu()
        )
        output_tensor = Tensor(
            dtype=dtype,
            shape=x.shape,
            device=x.device,
            requires_grad=requires_grad,
        )
        num_elements = shape_size(x.shape)
        cuda_kernel.run(
            (128, 1, 1),
            (128, 1, 1),
            [
                np.array(num_elements, dtype=np.int32),
                x,
                output_tensor,
            ],
        )
    elif x.device.type == "cpu":
        output_tensor = Tensor(
            dtype=dtype,
            shape=x.shape,
            device=x.device,
            requires_grad=requires_grad,
        )
        output_tensor.cpu_array = x.cpu_array.astype(dtype)
    else:
        raise InvalidDeviceError(x.device.type)
    if requires_grad:
        DAGTracker.instance().add_node("cast", [x, dtype], [output_tensor])

    return output_tensor


@DAGTracker.instance().register_backward_function("cast")
def backward(output_grad, x, dtype):
    x_grad = Tensor(dtype=x.dtype, shape=x.shape, device=x.device)
    if output_grad.device.type == "cuda":
        func_name = f"cast_reference_{dtype_to_str(dtype)}_{dtype_to_str(x.dtype)}"
        cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
        cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
            "cast.cu", func_name, output_grad.device.index
        )
        num_elements = shape_size(output_grad.shape)
        cuda_kernel.run(
            (128, 1, 1),
            (128, 1, 1),
            [
                np.array(num_elements, dtype=np.int32),
                output_grad,
                x_grad,
            ],
        )
    elif output_grad.device.type == "cpu":
        x_grad.cpu_array = output_grad.cpu_array.astype(x.dtype)
    else:
        raise InvalidDeviceError(output_grad.device.type)
    return [x_grad]
