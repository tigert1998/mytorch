import numpy as np

from mytorch.backends.cuda.env import CudaEnv
from mytorch.backends.cuda.cublas_lt import CublasLt
from mytorch.dtype import float16, float32
from mytorch.backends.backend_dispatcher import BackendDispatcher
from mytorch.tensor import Tensor


def _cuda_bmm(x: Tensor, y: Tensor, x_t: bool, y_t: bool, requires_grad: bool):
    from mytorch.tensor import Tensor, InvalidDataTypeError

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
        int(x._native().ptr),
        a_desc,
        int(y._native().ptr),
        b_desc,
        beta,
        int(z._native().ptr),
        d_desc,
        int(z._native().ptr),
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


@BackendDispatcher.instance().register_backend_function("cuda", "mm")
def cuda_mm(x, y):
    new_x = Tensor(tensor=x)
    new_x.shape = (1, *new_x.shape)
    new_y = Tensor(tensor=y)
    new_y.shape = (1, *new_y.shape)
    z = _cuda_bmm(new_x, new_y, False, False, False)
    z.shape = z.shape[1:]
    return z


@BackendDispatcher.instance().register_backend_function("cuda", "mm_backward")
def cuda_mm_backward(output_grad, x, y):
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
    return x_grad, y_grad


@BackendDispatcher.instance().register_backend_function("cuda", "bmm")
def cuda_bmm(x, y):
    return _cuda_bmm(x, y, False, False, False)


@BackendDispatcher.instance().register_backend_function("cuda", "bmm_backward")
def cuda_bmm_backward(output_grad, x, y):
    x_grad = _cuda_bmm(output_grad, y, False, True, False)
    y_grad = _cuda_bmm(x, output_grad, True, False, False)
    return x_grad, y_grad
