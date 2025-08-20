import numpy as np

from nn.module import Module
from nn.parameter import Parameter
from tensor import Tensor
from basic_ops import mm, reshape, permute, add


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device="cpu", dtype=np.float32
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.register_parameter(
            "weight",
            Parameter(
                Tensor(
                    cpu_array=np.zeros(
                        (self.out_features, self.in_features),
                        dtype=dtype,
                    ),
                    device=device,
                )
            ),
        )

        if bias:
            self.register_parameter(
                "bias",
                Parameter(
                    Tensor(
                        cpu_array=np.zeros(
                            (self.out_features,),
                            dtype=dtype,
                        ),
                        device=device,
                    )
                ),
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        b = x.shape[:-1]
        x = reshape(x, (-1, self.in_features))
        x = mm(x, permute(self.weight, (1, 0)))
        if self.bias is not None:
            x = add(x, self.bias)
        return reshape(x, b + (self.out_features,))
