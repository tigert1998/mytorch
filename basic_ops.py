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


def _cuda_bmm(x, y, x_t: bool, y_t: bool, requires_grad):
    from tensor import Tensor

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
        data_type, x.shape[1], x.shape[2], x.shape[2]
    )
    b_desc = cublas_lt.matrix_layout_create(
        data_type, y.shape[1], y.shape[2], y.shape[2]
    )
    d_desc = cublas_lt.matrix_layout_create(
        data_type, x.shape[1], y.shape[2], y.shape[2]
    )
    for desc in [a_desc, b_desc, d_desc]:
        cublas_lt.matrix_layout_set_attribute(
            desc,
            cublas_lt.CUBLASLT_MATRIX_LAYOUT_ORDER,
            np.array(cublas_lt.CUBLASLT_ORDER_ROW, np.int32),
        )
        cublas_lt.matrix_layout_set_attribute(
            desc,
            cublas_lt.CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            np.array(x.shape[0], np.int32),
        )

    alpha = np.array(1, dtype=x.dtype)
    beta = np.array(0, dtype=x.dtype)
    handle = cublas_lt.create()
    matmul_desc = cublas_lt.matmul_desc_create(compute_type, data_type)
    if x_t:
        cublas_lt.matmul_desc_set_attribute(
            matmul_desc,
            cublas_lt.CUBLASLT_MATMUL_DESC_TRANSA,
            np.array(cublas_lt.CUBLAS_OP_T, np.int32),
        )
    if y_t:
        cublas_lt.matmul_desc_set_attribute(
            matmul_desc,
            cublas_lt.CUBLASLT_MATMUL_DESC_TRANSB,
            np.array(cublas_lt.CUBLAS_OP_T, np.int32),
        )

    z = Tensor(
        shape=(x.shape[0], x.shape[1], y.shape[2]),
        dtype=x.dtype,
        device=x.device,
        requires_grad=requires_grad,
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

    for desc in [a_desc, b_desc, d_desc]:
        cublas_lt.matrix_layout_destroy(desc)
    cublas_lt.matmul_desc_destroy(matmul_desc)
    cublas_lt.destroy(handle)

    return z


def mm(x, y):
    from tensor import Tensor

    assert x.device == y.device

    if x.device.type == "cuda":
        new_x = Tensor(tensor=x)
        new_x.shape = (1, *new_x.shape)
        new_y = Tensor(tensor=y)
        new_y.shape = (1, *new_y.shape)
        z = _cuda_bmm(new_x, new_y, False, False, True)
        z.shape = z.shape[1:]

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
        new_x = Tensor(tensor=x)
        new_x.shape = (1, *new_x.shape)
        new_y = Tensor(tensor=y)
        new_y.shape = (1, *new_y.shape)
        new_output_grad = Tensor(tensor=output_grad)
        new_output_grad.shape = (1, *new_output_grad.shape)
        x_grad = _cuda_bmm(new_output_grad, new_y, False, True, False)
        y_grad = _cuda_bmm(new_x, new_output_grad, True, False, False)
        x_grad.shape = x_grad.shape[1:]
        y_grad.shape = y_grad.shape[1:]

    elif x.device.type == "cpu":
        x_grad_cpu_array = np.matmul(output_grad.cpu_array, y.cpu_array.T)
        y_grad_cpu_array = np.matmul(x.cpu_array.T, output_grad.cpu_array)
        x_grad = Tensor(cpu_array=x_grad_cpu_array, device="cpu")
        y_grad = Tensor(cpu_array=y_grad_cpu_array, device="cpu")

    else:
        raise InvalidDeviceError(x.device.type)

    return [x_grad, y_grad]


def bmm(x, y):
    from tensor import Tensor

    assert x.device == y.device

    if x.device.type == "cuda":
        z = _cuda_bmm(x, y, False, False, True)

    elif x.device.type == "cpu":
        z_cpu_array = np.matmul(x.cpu_array, y.cpu_array)
        z = Tensor(cpu_array=z_cpu_array, device="cpu", requires_grad=True)

    else:
        raise InvalidDeviceError(x.device.type)

    DAGTracker.instance().add_node("bmm", [x, y], [z])

    return z


@DAGTracker.instance().register_backward_function("bmm")
def bmm_backward(output_grad, x, y):
    from tensor import Tensor

    assert output_grad.device == x.device and output_grad.device == y.device

    if x.device.type == "cuda":
        x_grad = _cuda_bmm(output_grad, y, False, True, False)
        y_grad = _cuda_bmm(x, output_grad, True, False, False)

    elif x.device.type == "cpu":
        x_grad_cpu_array = np.matmul(output_grad.cpu_array, y.cpu_array.T)
        y_grad_cpu_array = np.matmul(x.cpu_array.T, output_grad.cpu_array)
        x_grad = Tensor(cpu_array=x_grad_cpu_array, device="cpu")
        y_grad = Tensor(cpu_array=y_grad_cpu_array, device="cpu")

    else:
        raise InvalidDeviceError(x.device.type)

    return [x_grad, y_grad]


def permute(x, permute_array):
    permute_array = [(i + len(permute_array) if i < 0 else i) for i in permute_array]

    from tensor import Tensor, CudaMemory

    if x.device.type == "cuda":
        if x.dtype == np.float32:
            func_name = "permute_reference_fp32"
        elif x.dtype == np.float16:
            func_name = "permute_reference_fp16"
        else:
            raise InvalidDataTypeError(x.dtype)
        cuda_kernel_and_stream_manager = CudaKernelAndStreamManager.instance()
        cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
            "basic_ops.cu", func_name, x.device.index
        )
        output_tensor = Tensor(
            dtype=x.dtype,
            device=x.device,
            shape=tuple([x.shape[i] for i in permute_array]),
            requires_grad=True,
        )

        num_bytes = np.dtype(np.int32).itemsize * len(permute_array)
        cuda_mem = CudaMemory(num_bytes * 2)
        cuda_mem.write(np.array(list(x.shape) + permute_array, dtype=np.int32))
        shape_ptr = int(cuda_mem.ptr)
        permute_array_ptr = shape_ptr + num_bytes

        num_elements = np.prod(x.shape)
        cuda_kernel.run(
            ((num_elements + 255) // 256, 1, 1),
            (256, 1, 1),
            [
                np.array(num_elements, np.int32),
                x,
                output_tensor,
                np.array(len(permute_array), np.int32),
                np.array(permute_array_ptr, dtype=np.uint64),
                np.array(shape_ptr, dtype=np.uint64),
            ],
        )

    elif x.device.type == "cpu":
        output_tensor = Tensor(
            cpu_array=np.transpose(x.cpu_array, permute_array),
            dtype=x.dtype,
            device=x.device,
            requires_grad=True,
        )

    else:
        raise InvalidDeviceError(x.device.type)

    DAGTracker.instance().add_node("permute", [x, permute_array], [output_tensor])

    return output_tensor


@DAGTracker.instance().register_backward_function("permute")
def permute_backward(output_grad, x, permute_array):
    from tensor import Tensor, CudaMemory

    permute_array = [(i + len(permute_array) if i < 0 else i) for i in permute_array]

    if x.device.type == "cuda":
        if x.dtype == np.float32:
            func_name = "permute_backward_reference_fp32"
        elif x.dtype == np.float16:
            func_name = "permute_backward_reference_fp16"
        else:
            raise InvalidDataTypeError(x.dtype)
        cuda_kernel_and_stream_manager = CudaKernelAndStreamManager.instance()
        cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
            "basic_ops.cu", func_name, x.device.index
        )
        input_grad = Tensor(
            dtype=x.dtype,
            device=x.device,
            shape=x.shape,
        )

        num_bytes = np.dtype(np.int32).itemsize * len(permute_array)
        cuda_mem = CudaMemory(num_bytes * 2)
        cuda_mem.write(np.array(list(x.shape) + permute_array, dtype=np.int32))
        shape_ptr = int(cuda_mem.ptr)
        permute_array_ptr = shape_ptr + num_bytes

        num_elements = np.prod(x.shape)
        cuda_kernel.run(
            ((num_elements + 255) // 256, 1, 1),
            (256, 1, 1),
            [
                np.array(num_elements, dtype=np.int32),
                np.array(len(permute_array), dtype=np.int32),
                np.array(permute_array_ptr, dtype=np.uint64),
                np.array(shape_ptr, dtype=np.uint64),
                input_grad,
                output_grad,
            ],
        )

    elif x.device.type == "cpu":
        reverse_permute_array = [None for _ in range(len(permute_array))]
        # [1, 2, 0] => [2, 0, 1]
        for i, permute in enumerate(permute_array):
            reverse_permute_array[permute] = i
        input_grad = Tensor(
            cpu_array=np.transpose(output_grad.cpu_array, reverse_permute_array),
            dtype=x.dtype,
            device=x.device,
            shape=x.shape,
        )

    else:
        raise InvalidDeviceError(x.device.type)

    return [input_grad]
