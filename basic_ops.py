import numpy as np
from tensor import InvalidDataTypeError, InvalidDeviceError
from cuda_utils import CudaKernelAndStreamManager
from autograd import DAGTracker


def sum(tensor):
    from tensor import Tensor

    dag_tracker = DAGTracker.instance()

    output_tensor = Tensor(dtype=tensor.dtype, shape=(1,), device=tensor.device)

    if tensor.device.type == "cuda":
        if tensor.dtype == np.float32:
            func_name = "sum_reference_fp32"
        elif tensor.dtype == np.float16:
            func_name = "sum_reference_fp16"
        else:
            raise InvalidDataTypeError(tensor.dtype)
        cuda_kernel_and_stream_manager = CudaKernelAndStreamManager.instance()
        cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
            "basic_ops.cu", func_name, tensor.device.index
        )
        num_elements = np.prod(tensor.shape)
        cuda_kernel.run(
            ((num_elements + 31) // 32, 1, 1),
            (32, 1, 1),
            [np.array(num_elements), tensor, output_tensor],
        )

    elif tensor.device.type == "cpu":
        ...

    else:
        raise InvalidDeviceError(tensor.device.type)

    dag_tracker.add_node("sum", [tensor], [output_tensor])

    return output_tensor


@DAGTracker.instance().register_backward_function("sum")
def sum_backward(output_grad, tensor):
    from tensor import Tensor

    tensor_grad = Tensor(shape=tensor.shape, dtype=tensor.dtype, device=tensor.device)

    if tensor.device.type == "cuda":
        if tensor.dtype == np.float32:
            func_name = "sum_backward_reference_fp32"
        elif tensor.dtype == np.float16:
            func_name = "sum_backward_reference_fp16"
        else:
            raise InvalidDataTypeError(tensor.dtype)
        cuda_kernel_and_stream_manager = CudaKernelAndStreamManager.instance()
        cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
            "basic_ops.cu", func_name, tensor.device.index
        )
        num_elements = np.prod(tensor.shape)
        cuda_kernel.run(
            ((num_elements + 31) // 32, 1, 1),
            (32, 1, 1),
            [np.array(num_elements), tensor, tensor_grad, output_grad],
        )

    elif tensor.device.type == "cpu":
        ...

    else:
        raise InvalidDeviceError(tensor.device.type)

    return [tensor_grad]
