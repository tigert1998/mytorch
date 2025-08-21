import numpy as np

from optim.optimizer import Optimizer
from tensor import Tensor, InvalidDataTypeError, InvalidDeviceError
from cuda.cuda_utils import CudaKernelAndStreamManager


class SGD(Optimizer):
    def __init__(
        self,
        params,
        lr=0.001,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        maximize=False,
    ):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "maximize": maximize,
        }

        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for param in group["params"]:
                self.state.setdefault(param, {})
                is_first_time = self.state[param].get("momentum_buffer") is None
                self.state[param].setdefault(
                    "momentum_buffer",
                    Tensor(shape=param.shape, device=param.device, dtype=param.dtype),
                )

                momentum_buffer = self.state[param]["momentum_buffer"]

                if param.device.type == "cuda":
                    if param.dtype == np.float32:
                        func_name = "sgd_reference_fp32"
                    elif param.dtype == np.float16:
                        func_name = "sgd_reference_fp16"
                    else:
                        raise InvalidDataTypeError(param.dtype)
                    cuda_kernel_and_stream_manager = (
                        CudaKernelAndStreamManager.instance()
                    )
                    cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
                        "optim.cu", func_name, param.device.index
                    )
                    num_elements = np.prod(param.shape)
                    cuda_kernel.run(
                        ((num_elements + 255) // 256, 1, 1),
                        (256, 1, 1),
                        [
                            np.array(int(is_first_time), dtype=np.int8),
                            np.array(num_elements, dtype=np.int32),
                            param,
                            param.grad,
                            momentum_buffer,
                            np.array(group["lr"], dtype=param.dtype),
                            np.array(group["weight_decay"], dtype=param.dtype),
                            np.array(group["momentum"], dtype=param.dtype),
                            np.array(group["dampening"], dtype=param.dtype),
                            np.array(int(group["nesterov"]), dtype=np.int8),
                            np.array(int(group["maximize"]), dtype=np.int8),
                        ],
                    )

                elif param.device.type == "cpu":
                    ...

                else:
                    raise InvalidDeviceError(param.device.type)
