import numpy as np
import math

import mytorch
from mytorch.nn.module import Module
from mytorch.nn.parameter import Parameter
from mytorch.nn.functional.conv import conv2d
from mytorch.nn.init import kaiming_uniform_, _calculate_fan_in_and_fan_out


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        device="cpu",
        dtype=mytorch.float32,
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
                mytorch.tensor(
                    data=np.zeros(
                        (self.out_channels, self.in_channels, *self.kernel_size),
                    ),
                    device=device,
                    dtype=dtype,
                )
            ),
        )
        if bias:
            self.register_parameter(
                "bias",
                Parameter(
                    mytorch.tensor(
                        data=np.zeros(
                            (self.out_channels,),
                        ),
                        dtype=dtype,
                        device=device,
                    )
                ),
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                self.bias.uniform_(-bound, bound)

    def forward(self, input):
        return conv2d(input, self.weight, self.bias, self.stride, self.padding)
