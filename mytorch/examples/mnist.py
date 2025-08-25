import numpy as np

from mytorch.nn.conv import Conv2d
from mytorch.nn.relu import ReLU
from mytorch.nn.linear import Linear
from mytorch.nn.module import Module
from mytorch.nn.sequential import Sequential
from mytorch.nn.max_pool import MaxPool2d
from mytorch.utils.data.mnist_dataset import MNISTDataset
from mytorch.utils.data.data_loader import DataLoader
from mytorch.nn.functional.cross_entropy import cross_entropy
from mytorch.optim.sgd import SGD
from mytorch.tensor import Tensor


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


if __name__ == "__main__":
    train_dataset = MNISTDataset("./datasets", True)
    test_dataset = MNISTDataset("./datasets", False)
    train_data_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = LeNet()
    model.to("cuda:0")
    optimizer = SGD(model.parameters())

    for epoch in range(50):
        for i, (x, y) in enumerate(train_data_loader):
            input_cpu_array = (
                (x.cpu_array.reshape((-1, 1, 28, 28)) / 255.0 - 0.1307) / 0.3081
            ).astype(np.float32)
            input_tensor = Tensor(cpu_array=input_cpu_array, device="cuda:0")
            logits = model(input_tensor)
            loss = cross_entropy(logits, y.to("cuda:0"))
            optimizer.zero_grad()
            loss.backward()
            if i % 16 == 0:
                print(
                    f"Epoch #{epoch} step #{i} loss: {loss.to("cpu").cpu_array.item()}"
                )
            optimizer.step()
