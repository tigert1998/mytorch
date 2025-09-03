import numpy as np

from mytorch.tensor import (
    InvalidDataTypeError,
    InvalidDeviceError,
    MismatchDevicesError,
    CudaMemory,
    shape_size,
    Tensor,
)
from mytorch.dtype import float16, float32
from mytorch.cuda.env import CudaEnv
from mytorch.autograd import DAGTracker
from mytorch.cuda.cublas_lt import CublasLt
from typing import Tuple


def _cuda_bmm(x: Tensor, y: Tensor, x_t: bool, y_t: bool, requires_grad: bool):
    cublas_lt = CublasLt.instance()

    if x.dtype == float32:
        compute_type = cublas_lt.CUBLAS_COMPUTE_32F
        data_type = cublas_lt.CUDA_R_32F
    elif x.dtype == float16:
        compute_type = cublas_lt.CUBLAS_COMPUTE_16F
        data_type = cublas_lt.CUDA_R_16F
    else:
        raise InvalidDataTypeError(x.dtype)

    cuda_context_manager = CudaEnv.instance().context_manager
    cuda_context_manager.set_device(x.device.index)
    cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
    stream = cuda_kernel_and_stream_manager.get_stream(x.device.index)

    a_desc = cublas_lt.matrix_layout_create(
        data_type, x.shape[1], x.shape[2], x.shape[2]
    )
    b_desc = cublas_lt.matrix_layout_create(
        data_type, y.shape[1], y.shape[2], y.shape[2]
    )
    z_rows = x.shape[2] if x_t else x.shape[1]
    z_cols = y.shape[1] if y_t else y.shape[2]
    d_desc = cublas_lt.matrix_layout_create(data_type, z_rows, z_cols, z_cols)
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

    alpha = np.array(1, dtype=x.dtype.np_dtype)
    beta = np.array(0, dtype=x.dtype.np_dtype)
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
        shape=(x.shape[0], z_rows, z_cols),
        dtype=x.dtype,
        device=x.device,
        requires_grad=requires_grad,
    )

    if x.device.type != "cuda" or y.device.type != "cuda" or z.device.type != "cuda":
        raise RuntimeError(
            f"Cannot run cuda_mm on non-GPU memory: {x.device}, {y.device}"
        )

    cublas_lt.matmul(
        handle,
        matmul_desc,
        alpha,
        int(x._cuda_ptr()),
        a_desc,
        int(y._cuda_ptr()),
        b_desc,
        beta,
        int(z._cuda_ptr()),
        d_desc,
        int(z._cuda_ptr()),
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


def mm(x: Tensor, y: Tensor):
    if x.device != y.device:
        raise MismatchDevicesError([x.device, y.device])

    requires_grad = x.requires_grad or y.requires_grad

    if x.device.type == "cuda":
        new_x = Tensor(tensor=x)
        new_x.shape = (1, *new_x.shape)
        new_y = Tensor(tensor=y)
        new_y.shape = (1, *new_y.shape)
        z = _cuda_bmm(new_x, new_y, False, False, requires_grad)
        z.shape = z.shape[1:]

    elif x.device.type == "cpu":
        z_cpu_array = np.matmul(x._numpy(), y._numpy())
        z = Tensor(cpu_array=z_cpu_array, device="cpu", requires_grad=requires_grad)

    else:
        raise InvalidDeviceError(x.device.type)

    if requires_grad:
        DAGTracker.instance().add_node("mm", [x, y], [z])

    return z


@DAGTracker.instance().register_backward_function("mm")
def mm_backward(output_grad: Tensor, x: Tensor, y: Tensor):
    from mytorch.tensor import Tensor

    if not (output_grad.device == x.device and output_grad.device == y.device):
        raise MismatchDevicesError([output_grad.device, x.device, y.device])

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
        x_grad_cpu_array = np.matmul(output_grad._numpy(), y._numpy().T)
        y_grad_cpu_array = np.matmul(x._numpy().T, output_grad._numpy())
        x_grad = Tensor(cpu_array=x_grad_cpu_array, device="cpu")
        y_grad = Tensor(cpu_array=y_grad_cpu_array, device="cpu")

    else:
        raise InvalidDeviceError(x.device.type)

    return [x_grad, y_grad]


def bmm(x: Tensor, y: Tensor):
    if x.device != y.device:
        raise MismatchDevicesError([x.device, y.device])

    requires_grad = x.requires_grad or y.requires_grad

    if x.device.type == "cuda":
        z = _cuda_bmm(x, y, False, False, requires_grad)

    elif x.device.type == "cpu":
        z_cpu_array = np.matmul(x._numpy(), y._numpy())
        z = Tensor(cpu_array=z_cpu_array, device="cpu", requires_grad=requires_grad)

    else:
        raise InvalidDeviceError(x.device.type)

    if requires_grad:
        DAGTracker.instance().add_node("bmm", [x, y], [z])

    return z


@DAGTracker.instance().register_backward_function("bmm")
def bmm_backward(output_grad: Tensor, x: Tensor, y: Tensor):
    if not (output_grad.device == x.device and output_grad.device == y.device):
        raise MismatchDevicesError([output_grad.device, x.device, y.device])

    if x.device.type == "cuda":
        x_grad = _cuda_bmm(output_grad, y, False, True, False)
        y_grad = _cuda_bmm(x, output_grad, True, False, False)

    elif x.device.type == "cpu":
        x_grad_cpu_array = np.matmul(output_grad._numpy(), y._numpy().T)
        y_grad_cpu_array = np.matmul(x._numpy().T, output_grad._numpy())
        x_grad = Tensor(cpu_array=x_grad_cpu_array, device="cpu")
        y_grad = Tensor(cpu_array=y_grad_cpu_array, device="cpu")

    else:
        raise InvalidDeviceError(x.device.type)

    return [x_grad, y_grad]


def permute(x: Tensor, dims: Tuple[int, ...]):
    if len(dims) != len(x.shape):
        raise RuntimeError(f"permute dims is invalid: {dims}")
    dims = tuple([(i + len(dims) if i < 0 else i) for i in dims])

    requires_grad = x.requires_grad

    if x.device.type == "cuda":
        if x.dtype == float32:
            func_name = "permute_reference_fp32"
        elif x.dtype == float16:
            func_name = "permute_reference_fp16"
        else:
            raise InvalidDataTypeError(x.dtype)
        cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
        cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
            "basic_ops.cu", func_name, x.device.index
        )
        output_tensor = Tensor(
            dtype=x.dtype,
            device=x.device,
            shape=tuple([x.shape[i] for i in dims]),
            requires_grad=requires_grad,
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

    elif x.device.type == "cpu":
        output_tensor = Tensor(
            cpu_array=np.transpose(x._numpy(), dims),
            dtype=x.dtype,
            device=x.device,
            requires_grad=requires_grad,
        )

    else:
        raise InvalidDeviceError(x.device.type)

    if requires_grad:
        DAGTracker.instance().add_node("permute", [x, dims], [output_tensor])

    return output_tensor


@DAGTracker.instance().register_backward_function("permute")
def permute_backward(output_grad: Tensor, x: Tensor, dims: Tuple[int, ...]):
    dims = tuple([(i + len(dims) if i < 0 else i) for i in dims])

    if x.device.type == "cuda":
        if x.dtype == float32:
            func_name = "permute_backward_reference_fp32"
        elif x.dtype == float16:
            func_name = "permute_backward_reference_fp16"
        else:
            raise InvalidDataTypeError(x.dtype)
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

    elif x.device.type == "cpu":
        reverse_dims = [-1 for _ in range(len(dims))]
        # [1, 2, 0] => [2, 0, 1]
        for i, permute in enumerate(dims):
            reverse_dims[permute] = i
        for i in reverse_dims:
            if i < 0:
                raise RuntimeError(f"Invalid dimension: {dims}")
        input_grad = Tensor(
            cpu_array=np.transpose(output_grad._numpy(), tuple(reverse_dims)),
            dtype=x.dtype,
            device=x.device,
            shape=x.shape,
        )

    else:
        raise InvalidDeviceError(x.device.type)

    return [input_grad]


def _calculate_reshaped_shape(
    original_shape: Tuple[int, ...], target_shape: Tuple[int, ...]
):
    total_elements = shape_size(original_shape)
    target_elements = 1
    unknown_dim_index = None

    for i, dim in enumerate(target_shape):
        if dim == -1:
            if unknown_dim_index is not None:
                raise ValueError("can only specify one unknown dimension")
            unknown_dim_index = i
        else:
            if dim <= 0:
                raise ValueError("negative dimensions not allowed except -1")
            target_elements *= dim

    if unknown_dim_index is not None:
        if total_elements % target_elements != 0:
            raise ValueError(
                f"cannot reshape array of size {total_elements} into shape {target_shape}"
            )
        unknown_dim = total_elements // target_elements
        target_shape_ls = list(target_shape)
        target_shape_ls[unknown_dim_index] = unknown_dim
        return tuple(target_shape_ls)
    else:
        if total_elements != target_elements:
            raise ValueError(
                f"cannot reshape array of size {total_elements} into shape {target_shape}"
            )
        return target_shape


def reshape(x: Tensor, shape: Tuple[int, ...]):
    requires_grad = x.requires_grad
    new_x = Tensor(tensor=x, requires_grad=requires_grad)
    new_x.shape = _calculate_reshaped_shape(x.shape, shape)

    if new_x.device.type == "cpu":
        new_x.cpu_array = new_x._numpy().reshape(new_x.shape)

    if requires_grad:
        DAGTracker.instance().add_node("reshape", [x, shape], [new_x])

    return new_x


@DAGTracker.instance().register_backward_function("reshape")
def reshape_backward(output_grad: Tensor, x: Tensor, shape: Tuple[int, ...]):
    input_grad = Tensor(tensor=output_grad)
    input_grad.shape = x.shape

    return [input_grad]
