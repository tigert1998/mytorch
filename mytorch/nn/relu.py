from mytorch.nn.module import Module
from mytorch.nn.functional.relu import relu


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return relu(x)
