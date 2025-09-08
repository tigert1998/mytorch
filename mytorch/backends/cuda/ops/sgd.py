import numpy as np

from mytorch.backends.cuda.env import CudaEnv
from mytorch.tensor import shape_size, Tensor
from mytorch.backends.backend_dispatcher import BackendDispatcher


@BackendDispatcher.instance().register_backend_function("cuda", "sgd")
def cuda_sgd(param: Tensor, momentum_buffer: Tensor, is_first_time: bool, lr, weight_decay, momentum, dampening,
             nesterov: bool, maximize: bool):
    func_name = f"sgd_reference_{param.dtype.name}"
    cuda_kernel_and_stream_manager = (
        CudaEnv.instance().kernel_and_stream_manager
    )
    cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
        "optim.cu", func_name, param.device.index
    )
    num_elements = shape_size(param.shape)
    cuda_kernel.run(
        ((num_elements + 255) // 256, 1, 1),
        (256, 1, 1),
        [
            np.array(int(is_first_time), dtype=np.int8),
            np.array(num_elements, dtype=np.int32),
            param,
            param.grad,
            momentum_buffer,
            np.array(lr, dtype=param.dtype.np_dtype),
            np.array(weight_decay, dtype=param.dtype.np_dtype),
            np.array(momentum, dtype=param.dtype.np_dtype),
            np.array(dampening, dtype=param.dtype.np_dtype),
            np.array(int(nesterov), dtype=np.int8),
            np.array(int(maximize), dtype=np.int8),
        ],
    )
