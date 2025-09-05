import numpy as np

from mytorch.backends.cuda.env import CudaEnv
from mytorch.backends.backend_dispatcher import BackendDispatcher


def _elementwise_op(name: str, arg_list: list, no_grad_and_inplace: bool, x):
    from mytorch.tensor import Tensor, shape_size

    func_name = f"{name}_reference_{x.dtype.name}"
    cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
    cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
        "elementwise_ops.cu",
        func_name,
        x.device.index,
    )
    if no_grad_and_inplace:
        output_tensor = np.array(0, np.uint64)
    else:
        output_tensor = Tensor(
            dtype=x.dtype,
            shape=x.shape,
            device=x.device,
        )
    num_elements = shape_size(x.shape)
    cuda_kernel.run(
        (128, 1, 1),
        (128, 1, 1),
        [
            np.array(num_elements, dtype=np.int32),
            x,
            *arg_list,
            output_tensor,
        ],
    )

    if not no_grad_and_inplace:
        return output_tensor


def _elementwise_op_backward(name, output_grad, x, args):
    from mytorch.tensor import Tensor, shape_size

    x_grad = Tensor(dtype=x.dtype, shape=x.shape, device=x.device)
    x_grad.fill_(0)

    func_name = f"{name}_backward_reference_{output_grad.dtype.name}"
    cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
    cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
        "elementwise_ops.cu", func_name, output_grad.device.index
    )
    num_elements = shape_size(output_grad.shape)
    cuda_kernel.run(
        (128, 1, 1),
        (128, 1, 1),
        [
            np.array(num_elements, dtype=np.int32),
            x,
            *args,
            x_grad,
            output_grad,
        ],
    )

    return [x_grad]


@BackendDispatcher.instance().register_backend_function("cuda", "fill")
def _fill(x, value):
    _elementwise_op("fill", [np.array(value, dtype=x.dtype.np_dtype)], True, x)


@BackendDispatcher.instance().register_backend_function("cuda", "normal")
def _normal(x, seed, mean, stddev):
    _elementwise_op(
        "normal",
        [
            np.array(seed, dtype=np.uint64),
            np.array(mean, dtype=x.dtype.np_dtype),
            np.array(stddev, dtype=x.dtype.np_dtype),
        ],
        True,
        x,
    )


@BackendDispatcher.instance().register_backend_function("cuda", "uniform")
def _uniform(x, seed, a, b):
    _elementwise_op(
        "uniform",
        [
            np.array(seed, dtype=np.uint64),
            np.array(a, dtype=x.dtype.np_dtype),
            np.array(b, dtype=x.dtype.np_dtype),
        ],
        True,
        x,
    )


@BackendDispatcher.instance().register_backend_function("cuda", "relu")
def _relu(x):
    return _elementwise_op("relu", [], False, x)


@BackendDispatcher.instance().register_backend_function("cuda", "relu_backward")
def _relu_backward(output_grad, x):
    return _elementwise_op_backward("relu", output_grad, x, [])
