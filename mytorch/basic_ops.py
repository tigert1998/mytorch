import numpy as np

from mytorch.tensor import (
    InvalidDataTypeError,
    InvalidDeviceError,
    CudaMemory,
    shape_size,
    Tensor,
)
from mytorch.cuda.env import CudaEnv
from mytorch.autograd import DAGTracker
from mytorch.cuda.cublas_lt import CublasLt


def _calculate_reduce_shape(shape, axis, keepdim):
    assert np.all([0 <= i and i < len(shape) and isinstance(i, int) for i in axis])
    if keepdim:
        shape = [(1 if i in axis else shape_i) for i, shape_i in enumerate(shape)]
    else:
        shape = [shape_i for i, shape_i in enumerate(shape) if i not in axis]
    return tuple(shape)


def sum(tensor, reduce_axis=None, keepdim=False):
    from mytorch.tensor import Tensor

    if reduce_axis is None:
        reduce_axis = tuple(range(len(tensor.shape)))
    reduce_axis = tuple(sorted(reduce_axis))
    output_shape = _calculate_reduce_shape(tensor.shape, reduce_axis, keepdim)

    dag_tracker = DAGTracker.instance()

    output_tensor = Tensor(
        dtype=tensor.dtype, shape=output_shape, device=tensor.device, requires_grad=True
    )
    output_tensor.fill_(0)

    if tensor.device.type == "cuda":
        if tensor.dtype == np.float32:
            func_name = "sum_reference_fp32"
        elif tensor.dtype == np.float16:
            func_name = "sum_reference_fp16"
        else:
            raise InvalidDataTypeError(tensor.dtype)
        cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
        cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
            "basic_ops.cu", func_name, tensor.device.index
        )

        tensor_shape_num_bytes = len(tensor.shape) * np.dtype(np.int32).itemsize
        reduce_axis_num_bytes = len(reduce_axis) * np.dtype(np.int32).itemsize
        if tensor_shape_num_bytes + reduce_axis_num_bytes > 0:
            cuda_mem = CudaMemory(tensor_shape_num_bytes + reduce_axis_num_bytes)
            cuda_mem.write(
                np.array(list(tensor.shape) + list(reduce_axis), dtype=np.int32)
            )
            tensor_shape_ptr = int(cuda_mem.ptr)
            reduce_axis_ptr = tensor_shape_ptr + tensor_shape_num_bytes
        else:
            tensor_shape_ptr = reduce_axis_ptr = 0

        num_elements = shape_size(tensor.shape)
        cuda_kernel.run(
            ((num_elements + 255) // 256, 1, 1),
            (256, 1, 1),
            [
                np.array(num_elements, np.int32),
                tensor,
                np.array(len(tensor.shape), np.int32),
                np.array(tensor_shape_ptr, np.uint64),
                np.array(len(reduce_axis), np.int32),
                np.array(reduce_axis_ptr, np.uint64),
                output_tensor,
            ],
        )

    elif tensor.device.type == "cpu":
        output_tensor.cpu_array = np.sum(
            tensor.cpu_array, axis=reduce_axis, keepdims=keepdim
        )

    else:
        raise InvalidDeviceError(tensor.device.type)

    dag_tracker.add_node("sum", [tensor, reduce_axis, keepdim], [output_tensor])

    return output_tensor


@DAGTracker.instance().register_backward_function("sum")
def sum_backward(output_grad, tensor, reduce_axis, keepdim):
    from mytorch.tensor import Tensor

    tensor_grad = Tensor(shape=tensor.shape, dtype=tensor.dtype, device=tensor.device)

    if tensor.device.type == "cuda":
        if tensor.dtype == np.float32:
            func_name = "sum_backward_reference_fp32"
        elif tensor.dtype == np.float16:
            func_name = "sum_backward_reference_fp16"
        else:
            raise InvalidDataTypeError(tensor.dtype)
        cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
        cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
            "basic_ops.cu", func_name, tensor.device.index
        )

        tensor_shape_num_bytes = len(tensor.shape) * np.dtype(np.int32).itemsize
        reduce_axis_num_bytes = len(reduce_axis) * np.dtype(np.int32).itemsize
        if tensor_shape_num_bytes + reduce_axis_num_bytes > 0:
            cuda_mem = CudaMemory(tensor_shape_num_bytes + reduce_axis_num_bytes)
            cuda_mem.write(
                np.array(list(tensor.shape) + list(reduce_axis), dtype=np.int32)
            )
            tensor_shape_ptr = int(cuda_mem.ptr)
            reduce_axis_ptr = tensor_shape_ptr + tensor_shape_num_bytes
        else:
            tensor_shape_ptr = reduce_axis_ptr = 0

        num_elements = shape_size(tensor.shape)
        cuda_kernel.run(
            ((num_elements + 255) // 256, 1, 1),
            (256, 1, 1),
            [
                np.array(num_elements),
                tensor,
                np.array(len(tensor.shape), np.int32),
                np.array(tensor_shape_ptr, np.uint64),
                np.array(len(reduce_axis), np.int32),
                np.array(reduce_axis_ptr, np.uint64),
                tensor_grad,
                output_grad,
            ],
        )

    elif tensor.device.type == "cpu":
        if keepdim:
            src = output_grad.cpu_array
        else:
            src = np.expand_dims(output_grad.cpu_array, reduce_axis)
        np.copyto(tensor_grad.cpu_array, src)

    else:
        raise InvalidDeviceError(tensor.device.type)

    return [tensor_grad]


def _cuda_bmm(x, y, x_t: bool, y_t: bool, requires_grad):
    from mytorch.tensor import Tensor

    cublas_lt = CublasLt.instance()

    if x.dtype == np.float32:
        compute_type = cublas_lt.CUBLAS_COMPUTE_32F
        data_type = cublas_lt.CUDA_R_32F
    elif x.dtype == np.float16:
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
        shape=(x.shape[0], z_rows, z_cols),
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
    from mytorch.tensor import Tensor

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
    from mytorch.tensor import Tensor

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
    from mytorch.tensor import Tensor

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
    from mytorch.tensor import Tensor

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

    if x.device.type == "cuda":
        if x.dtype == np.float32:
            func_name = "permute_reference_fp32"
        elif x.dtype == np.float16:
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
            shape=tuple([x.shape[i] for i in permute_array]),
            requires_grad=True,
        )

        num_bytes = np.dtype(np.int32).itemsize * len(permute_array)
        if num_bytes > 0:
            cuda_mem = CudaMemory(num_bytes * 2)
            cuda_mem.write(np.array(list(x.shape) + permute_array, dtype=np.int32))
            shape_ptr = int(cuda_mem.ptr)
            permute_array_ptr = shape_ptr + num_bytes
        else:
            shape_ptr = permute_array_ptr = 0

        num_elements = shape_size(x.shape)
        cuda_kernel.run(
            ((num_elements + 255) // 256, 1, 1),
            (256, 1, 1),
            [
                np.array(num_elements, np.int32),
                x,
                output_tensor,
                np.array(len(permute_array), np.int32),
                np.array(shape_ptr, dtype=np.uint64),
                np.array(permute_array_ptr, dtype=np.uint64),
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
    permute_array = [(i + len(permute_array) if i < 0 else i) for i in permute_array]

    if x.device.type == "cuda":
        if x.dtype == np.float32:
            func_name = "permute_backward_reference_fp32"
        elif x.dtype == np.float16:
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

        num_bytes = np.dtype(np.int32).itemsize * len(permute_array)
        if num_bytes > 0:
            cuda_mem = CudaMemory(num_bytes * 2)
            cuda_mem.write(np.array(list(x.shape) + permute_array, dtype=np.int32))
            shape_ptr = int(cuda_mem.ptr)
            permute_array_ptr = shape_ptr + num_bytes
        else:
            shape_ptr = permute_array_ptr = 0

        num_elements = shape_size(x.shape)
        cuda_kernel.run(
            ((num_elements + 255) // 256, 1, 1),
            (256, 1, 1),
            [
                np.array(num_elements, dtype=np.int32),
                np.array(len(permute_array), dtype=np.int32),
                np.array(shape_ptr, dtype=np.uint64),
                np.array(permute_array_ptr, dtype=np.uint64),
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


def _calculate_reshaped_shape(original_shape, target_shape):
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
        target_shape = list(target_shape)
        target_shape[unknown_dim_index] = unknown_dim
        return tuple(target_shape)
    else:
        if total_elements != target_elements:
            raise ValueError(
                f"cannot reshape array of size {total_elements} into shape {target_shape}"
            )
        return target_shape


def reshape(x, shape):
    from mytorch.tensor import Tensor

    new_x = Tensor(tensor=x, requires_grad=True)
    new_x.shape = _calculate_reshaped_shape(x.shape, shape)

    DAGTracker.instance().add_node("reshape", [x, shape], [new_x])

    return new_x


@DAGTracker.instance().register_backward_function("reshape")
def reshape_backward(output_grad, x, shape):
    from mytorch.tensor import Tensor

    input_grad = Tensor(tensor=output_grad)
    input_grad.shape = x.shape

    return [input_grad]
