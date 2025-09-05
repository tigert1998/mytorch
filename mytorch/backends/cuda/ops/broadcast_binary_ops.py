import numpy as np
import numpy.typing as npt

from mytorch.backends.cuda.env import CudaEnv
from mytorch.backends.backend_dispatcher import BackendDispatcher
from mytorch.backends.utils import calculate_broadcast_shape


def _broadcast_binary_op(name: str, arg_list: list, no_grad_and_inplace: bool, x, y):
    from mytorch.tensor import CudaMemory, shape_size, Tensor

    output_shape = calculate_broadcast_shape(x.shape, y.shape)
    func_name = f"{name}_reference_{x.dtype.name}"
    cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
    cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
        "broadcast_binary_ops.cu", func_name, x.device.index
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
    num_elements = shape_size(output_shape)
    if no_grad_and_inplace:
        null_ptr = np.array(0, np.uint64)
    else:
        output_tensor = Tensor(
            dtype=x.dtype,
            shape=output_shape,
            device=x.device,
        )
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
            *arg_list,
            null_ptr if no_grad_and_inplace else output_tensor,
        ],
    )
    if not no_grad_and_inplace:
        return output_tensor


def _broadcast_binary_op_backward(name, output_grad, x, y, args):
    from mytorch.tensor import CudaMemory, shape_size, Tensor

    x_grad = Tensor(dtype=x.dtype, shape=x.shape, device=x.device)
    x_grad.fill_(0)
    y_grad = Tensor(dtype=y.dtype, shape=y.shape, device=y.device)
    y_grad.fill_(0)

    func_name = f"{name}_backward_reference_{output_grad.dtype.name}"
    cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
    cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
        "broadcast_binary_ops.cu", func_name, output_grad.device.index
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
    num_elements = shape_size(output_grad.shape)
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
            *args,
            x_grad,
            y_grad,
            output_grad,
        ],
    )

    return x_grad, y_grad


@BackendDispatcher.instance().register_backend_function("cuda", "add")
def add(x, y, alpha):
    return _broadcast_binary_op(
        "add", [np.array(alpha, dtype=x.dtype.np_dtype)], False, x, y
    )


@BackendDispatcher.instance().register_backend_function("cuda", "add_backward")
def add_backward(output_grad, x, y, alpha):
    return _broadcast_binary_op_backward(
        "add", output_grad, x, y, [np.array(alpha, dtype=x.dtype.np_dtype)]
    )


@BackendDispatcher.instance().register_backend_function("cuda", "sub")
def sub(x, y, alpha):
    return _broadcast_binary_op(
        "sub", [np.array(alpha, dtype=x.dtype.np_dtype)], False, x, y
    )


@BackendDispatcher.instance().register_backend_function("cuda", "sub_backward")
def sub_backward(output_grad, x, y, alpha):
    return _broadcast_binary_op_backward(
        "sub", output_grad, x, y, [np.array(alpha, dtype=x.dtype.np_dtype)]
    )


@BackendDispatcher.instance().register_backend_function("cuda", "mul")
def mul(x, y):
    return _broadcast_binary_op("mul", [], False, x, y)


@BackendDispatcher.instance().register_backend_function("cuda", "mul_backward")
def mul_backward(output_grad, x, y):
    return _broadcast_binary_op_backward("mul", output_grad, x, y, [])


@BackendDispatcher.instance().register_backend_function("cuda", "div")
def div(x, y):
    return _broadcast_binary_op("div", [], False, x, y)


@BackendDispatcher.instance().register_backend_function("cuda", "div_backward")
def div_backward(output_grad, x, y):
    return _broadcast_binary_op_backward("div", output_grad, x, y, [])


@BackendDispatcher.instance().register_backend_function("cuda", "pow")
def pow(x, y):
    return _broadcast_binary_op("pow", [], False, x, y)


@BackendDispatcher.instance().register_backend_function("cuda", "pow_backward")
def pow_backward(output_grad, x, y):
    return _broadcast_binary_op_backward("pow", output_grad, x, y, [])


@BackendDispatcher.instance().register_backend_function("cuda", "copy")
def copy(x, y):
    _broadcast_binary_op("copy", [], True, x, y)
