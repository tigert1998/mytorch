import numpy as np

from mytorch.tensor import InvalidDataTypeError, Tensor
from mytorch.backends.cuda.env import CudaEnv, BackendDispatcher
from mytorch.dtype import int64


@BackendDispatcher.instance().register_backend_function("cuda", "cross_entropy")
def cuda_cross_entropy(input: Tensor, target: Tensor):
    if target.dtype != int64:
        raise InvalidDataTypeError(target.dtype)
    batch_size, num_classes = input.shape

    tensor = Tensor(
        dtype=input.dtype,
        shape=(1,),
        device=input.device,
    )
    tensor.fill_(0)

    func_name = f"cross_entropy_reference_{input.dtype.name}"
    cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
    cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
        "cross_entropy.cu", func_name, input.device.index
    )
    cuda_kernel.run(
        (32, 1, 1),
        (32, 1, 1),
        [
            np.array(batch_size),
            np.array(num_classes),
            input,
            target,
            tensor,
        ],
    )

    return tensor


@BackendDispatcher.instance().register_backend_function("cuda", "cross_entropy_backward")
def cuda_cross_entropy_backward(output_grad: Tensor, input: Tensor, target: Tensor):
    batch_size, num_classes = input.shape

    input_grad = Tensor(
        dtype=input.dtype,
        shape=input.shape,
        device=input.device,
    )

    func_name = f"cross_entropy_backward_reference_{input.dtype.name}"
    cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
    cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
        "cross_entropy.cu", func_name, input.device.index
    )
    cuda_kernel.run(
        (32, 1, 1),
        (32, 1, 1),
        [
            np.array(batch_size),
            np.array(num_classes),
            input,
            target,
            input_grad,
            output_grad,
        ],
    )

    return [input_grad]
