from mytorch.nn.conv import Conv2d
from mytorch.nn.relu import ReLU
from mytorch.nn.linear import Linear
from mytorch.nn.module import Module
from mytorch.nn.sequential import Sequential
from mytorch.nn.max_pool import MaxPool2d


class LeNet(Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Sequential(
            Conv2d(1, 6, kernel_size=5, padding=2),
            ReLU(),
            MaxPool2d(2, 2),
            Conv2d(6, 16, 5),
            ReLU(),
            MaxPool2d(2, 2),
        )
        self.layer2 = Sequential(
            Linear(16 * 5 * 5, 120),
            ReLU(),
            Linear(120, 84),
            ReLU(),
            Linear(84, 10),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = x.reshape((-1, 16 * 5 * 5))
        x = self.layer2(x)
        return x
