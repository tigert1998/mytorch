import numpy as np

from mytorch.optim.optimizer import Optimizer
from mytorch.tensor import Tensor, InvalidDeviceError, shape_size
from mytorch.cuda.env import CudaEnv


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
                            np.array(group["lr"], dtype=param.dtype.np_dtype),
                            np.array(group["weight_decay"], dtype=param.dtype.np_dtype),
                            np.array(group["momentum"], dtype=param.dtype.np_dtype),
                            np.array(group["dampening"], dtype=param.dtype.np_dtype),
                            np.array(int(group["nesterov"]), dtype=np.int8),
                            np.array(int(group["maximize"]), dtype=np.int8),
                        ],
                    )

                elif param.device.type == "cpu":
                    g = (
                        -param.grad.cpu_array
                        if group["maximize"]
                        else param.grad.cpu_array
                    )
                    g += group["weight_decay"] * param.cpu_array
                    momentum_buffer.cpu_array = (
                        g
                        if is_first_time
                        else (
                            group["momentum"] * momentum_buffer.cpu_array
                            + (1 - group["dampening"]) * g
                        )
                    )
                    g = (
                        (g + group["momentum"] * momentum_buffer.cpu_array)
                        if group["nesterov"]
                        else momentum_buffer.cpu_array
                    )
                    param.cpu_array -= g * group["lr"]

                else:
                    raise InvalidDeviceError(param.device.type)
