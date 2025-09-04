from functools import cache

import numpy as np

from mytorch.tensor import Tensor, shape_size, InvalidDeviceError
from mytorch.cuda.env import CudaEnv
from mytorch.autograd import DAGTracker
from mytorch.dtype import DType, int8, int16, int32, int64, float16, float32, float64


@cache
def _generate_cast_cu():
    dtypes = [int8, int16, int32, int64, float16, float32, float64]
    return CudaEnv.instance().compiler.get_templated_source(
        "cast.cu",
        {
            "cast_reference": [
                (s_dtype, t_dtype)
                for s_dtype in dtypes
                for t_dtype in dtypes
                if s_dtype != t_dtype
            ]
        },
    )


def _cast(x: Tensor, dtype: DType):
    if x.dtype == dtype:
        return x

    requires_grad = x.requires_grad and x.dtype.is_floating and dtype.is_floating

    if x.device.type == "cuda":
        cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
        func_name = f"cast_reference_{x.dtype.name}_{dtype.name}"
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
        output_tensor.cpu_array = x._numpy().astype(dtype.np_dtype)
    else:
        raise InvalidDeviceError(x.device.type)
    if requires_grad:
        DAGTracker.instance().add_node("cast", [x, dtype], [output_tensor])

    return output_tensor


@DAGTracker.instance().register_backward_function("cast")
def backward(output_grad: Tensor, x: Tensor, dtype: DType):
    x_grad = Tensor(dtype=x.dtype, shape=x.shape, device=x.device)
    if output_grad.device.type == "cuda":
        func_name = f"cast_reference_{dtype.name}_{x.dtype.name}"
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
        x_grad.cpu_array = output_grad._numpy().astype(x.dtype.np_dtype)
    else:
        raise InvalidDeviceError(output_grad.device.type)
    return [x_grad]
