import numpy as np

from mytorch.backends.cuda.env import CudaEnv
from mytorch.backends.backend_dispatcher import BackendDispatcher


@BackendDispatcher.instance().register_backend_function("cuda", "cast")
def cuda_cast(x, dtype):
    from mytorch.tensor import Tensor, shape_size

    cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
    func_name = f"cast_reference_{x.dtype.name}_{dtype.name}"
    cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
        "cast.cu", func_name, x.device.index
    )
    output_tensor = Tensor(
        dtype=dtype,
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
            output_tensor,
        ],
    )
    return output_tensor
