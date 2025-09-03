import numpy as np
import math

import mytorch
from mytorch.nn.module import Module
from mytorch.nn.parameter import Parameter
from mytorch.tensor import Tensor
from mytorch.ops.basic_ops import mm, reshape, permute
from mytorch.nn.init import kaiming_uniform_, _calculate_fan_in_and_fan_out


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device="cpu", dtype=mytorch.float32
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.register_parameter(
            "weight",
            Parameter(
                mytorch.tensor(
                    data=np.zeros(
                        (self.out_features, self.in_features),
                    ),
                    dtype=dtype,
                    device=device,
                )
            ),
        )

        if bias:
            self.register_parameter(
                "bias",
                Parameter(
                    mytorch.tensor(
                        data=np.zeros(
                            (self.out_features,),
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
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias.uniform_(-bound, bound)

    def forward(self, x):
        b = x.shape[:-1]
        x = reshape(x, (-1, self.in_features))
        x = mm(x, permute(self.weight, (1, 0)))
        if self.bias is not None:
            x = x + self.bias
        return reshape(x, b + (self.out_features,))
