import numpy as np
from nn.module import Module
from nn.parameter import Parameter, Tensor


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        padding,
        bias=True,
        device="cpu",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding

        self.register_parameter(
            "weight",
            Parameter(
                Tensor(
                    cpu_array=np.zeros(
                        (self.out_channels, self.in_channels, *self.kernel_size)
                    ),
                    device=device,
                )
            ),
        )
        if bias:
            self.register_parameter(
                "bias",
                Parameter(
                    Tensor(cpu_array=np.zeros((self.out_channels,)), device=device)
                ),
            )
        else:
            self.register_parameter("bias", None)
