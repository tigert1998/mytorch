from mytorch.nn.module import Module
from mytorch.nn.functional.pool_ops import max_pool2d


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()

        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding

    def forward(self, input):
        return max_pool2d(
            input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
