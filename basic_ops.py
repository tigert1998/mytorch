import numpy as np
from tensor import InvalidDataTypeError, InvalidDeviceError
from cuda.cuda_utils import (
    CudaKernelAndStreamManager,
    CublasLt,
    CudaContextManager,
)
from autograd import DAGTracker


def sum(tensor):
    from tensor import Tensor

    dag_tracker = DAGTracker.instance()

    output_tensor = Tensor(
        dtype=tensor.dtype, shape=(1,), device=tensor.device, requires_grad=True
    )

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
            ((num_elements + 255) // 256, 1, 1),
            (256, 1, 1),
            [np.array(num_elements), tensor, output_tensor],
        )

    elif tensor.device.type == "cpu":
        output_tensor.fill_(tensor.cpu_array.sum())

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
            ((num_elements + 255) // 256, 1, 1),
            (256, 1, 1),
            [np.array(num_elements), tensor, tensor_grad, output_grad],
        )

    elif tensor.device.type == "cpu":
        tensor_grad.fill_(output_grad.cpu_array.item())

    else:
        raise InvalidDeviceError(tensor.device.type)

    return [tensor_grad]


def mm(x, y):
    from tensor import Tensor

    assert x.device == y.device

    if x.device.type == "cuda":
        cublas_lt = CublasLt.instance()

        if x.dtype == np.float32:
            compute_type = cublas_lt.CUBLAS_COMPUTE_32F
            data_type = cublas_lt.CUDA_R_32F
        elif x.dtype == np.float16:
            compute_type = cublas_lt.CUBLAS_COMPUTE_16F
            data_type = cublas_lt.CUDA_R_16F
        else:
            raise InvalidDataTypeError(x.dtype)

        cuda_context_manager = CudaContextManager().instance()
        cuda_context_manager.set_device(x.device.index)
        cuda_kernel_and_stream_manager = CudaKernelAndStreamManager.instance()
        stream = cuda_kernel_and_stream_manager.get_stream(x.device.index)

        a_desc = cublas_lt.matrix_layout_create(
            data_type, x.shape[0], x.shape[1], x.shape[1]
        )
        b_desc = cublas_lt.matrix_layout_create(
            data_type, y.shape[0], y.shape[1], y.shape[1]
        )
        d_desc = cublas_lt.matrix_layout_create(
            data_type, x.shape[0], y.shape[1], y.shape[1]
        )
        for desc in [a_desc, b_desc, d_desc]:
            cublas_lt.matrix_layout_set_attribute(
                desc,
                cublas_lt.CUBLASLT_MATRIX_LAYOUT_ORDER,
                np.array(cublas_lt.CUBLASLT_ORDER_ROW, np.int32),
            )

        alpha = np.array(1, dtype=x.dtype)
        beta = np.array(0, dtype=x.dtype)
        handle = cublas_lt.create()
        matmul_desc = cublas_lt.matmul_desc_create(compute_type, data_type)

        z = Tensor(
            shape=(x.shape[0], y.shape[1]),
            dtype=x.dtype,
            device=x.device,
            requires_grad=True,
        )
        cublas_lt.matmul(
            handle,
            matmul_desc,
            alpha,
            int(x.cuda_ptr.ptr),
            a_desc,
            int(y.cuda_ptr.ptr),
            b_desc,
            beta,
            int(z.cuda_ptr.ptr),
            d_desc,
            int(z.cuda_ptr.ptr),
            d_desc,
            None,
            0,
            0,
            stream,
        )

    elif x.device.type == "cpu":
        z_cpu_array = np.matmul(x.cpu_array, y.cpu_array)
        z = Tensor(cpu_array=z_cpu_array, device="cpu", requires_grad=True)

    else:
        raise InvalidDeviceError(x.device.type)

    DAGTracker.instance().add_node("mm", [x, y], [z])

    return z


@DAGTracker.instance().register_backward_function("mm")
def mm_backward(output_grad, x, y):
    from tensor import Tensor

    assert output_grad.device == x.device and output_grad.device == y.device

    if x.device.type == "cuda":
        cublas_lt = CublasLt.instance()

        if x.dtype == np.float32:
            compute_type = cublas_lt.CUBLAS_COMPUTE_32F
            data_type = cublas_lt.CUDA_R_32F
        elif x.dtype == np.float16:
            compute_type = cublas_lt.CUBLAS_COMPUTE_16F
            data_type = cublas_lt.CUDA_R_16F
        else:
            raise InvalidDataTypeError(x.dtype)

        cuda_context_manager = CudaContextManager().instance()
        cuda_context_manager.set_device(x.device.index)
        cuda_kernel_and_stream_manager = CudaKernelAndStreamManager.instance()
        stream = cuda_kernel_and_stream_manager.get_stream(x.device.index)

        output_desc = cublas_lt.matrix_layout_create(
            data_type, x.shape[0], y.shape[1], y.shape[1]
        )
        x_desc = cublas_lt.matrix_layout_create(
            data_type, x.shape[0], x.shape[1], x.shape[1]
        )
        y_desc = cublas_lt.matrix_layout_create(
            data_type, y.shape[0], y.shape[1], y.shape[1]
        )
        for desc in [output_desc, x_desc, y_desc]:
            cublas_lt.matrix_layout_set_attribute(
                desc,
                cublas_lt.CUBLASLT_MATRIX_LAYOUT_ORDER,
                np.array(cublas_lt.CUBLASLT_ORDER_ROW, np.int32),
            )

        alpha = np.array(1, dtype=x.dtype)
        beta = np.array(0, dtype=x.dtype)

        handle = cublas_lt.create()

        x_grad_matmul_desc = cublas_lt.matmul_desc_create(compute_type, data_type)
        cublas_lt.matmul_desc_set_attribute(
            x_grad_matmul_desc,
            cublas_lt.CUBLASLT_MATMUL_DESC_TRANSB,
            np.array(cublas_lt.CUBLAS_OP_T, np.int32),
        )
        x_grad = Tensor(shape=(x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        cublas_lt.matmul(
            handle,
            x_grad_matmul_desc,
            alpha,
            int(output_grad.cuda_ptr.ptr),
            output_desc,
            int(y.cuda_ptr.ptr),
            y_desc,
            beta,
            int(x_grad.cuda_ptr.ptr),
            x_desc,
            int(x_grad.cuda_ptr.ptr),
            x_desc,
            None,
            0,
            0,
            stream,
        )

        y_grad_matmul_desc = cublas_lt.matmul_desc_create(compute_type, data_type)
        cublas_lt.matmul_desc_set_attribute(
            y_grad_matmul_desc,
            cublas_lt.CUBLASLT_MATMUL_DESC_TRANSA,
            np.array(cublas_lt.CUBLAS_OP_T, np.int32),
        )
        y_grad = Tensor(shape=(y.shape[0], y.shape[1]), dtype=y.dtype, device=y.device)
        cublas_lt.matmul(
            handle,
            y_grad_matmul_desc,
            alpha,
            int(x.cuda_ptr.ptr),
            x_desc,
            int(output_grad.cuda_ptr.ptr),
            output_desc,
            beta,
            int(y_grad.cuda_ptr.ptr),
            y_desc,
            int(y_grad.cuda_ptr.ptr),
            y_desc,
            None,
            0,
            0,
            stream,
        )

    elif x.device.type == "cpu":
        x_grad_cpu_array = np.matmul(output_grad.cpu_array, y.cpu_array.T)
        y_grad_cpu_array = np.matmul(x.cpu_array.T, output_grad.cpu_array)
        x_grad = Tensor(cpu_array=x_grad_cpu_array, device="cpu")
        y_grad = Tensor(cpu_array=y_grad_cpu_array, device="cpu")

    else:
        raise InvalidDeviceError(x.device.type)

    return [x_grad, y_grad]
