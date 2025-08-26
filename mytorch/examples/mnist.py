import numpy as np

from mytorch.nn.conv import Conv2d
from mytorch.nn.relu import ReLU
from mytorch.nn.linear import Linear
from mytorch.nn.module import Module
from mytorch.nn.sequential import Sequential
from mytorch.nn.batch_norm import BatchNorm2d
from mytorch.nn.max_pool import MaxPool2d
from mytorch.utils.data.mnist_dataset import MNISTDataset
from mytorch.utils.data.data_loader import DataLoader
from mytorch.nn.functional.cross_entropy import cross_entropy
from mytorch.optim.sgd import SGD
from mytorch.tensor import Tensor
from mytorch.autograd import no_grad


class CNN(Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Sequential(
            Conv2d(1, 32, kernel_size=3, padding=1),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(2, 2),
            Conv2d(32, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(2, 2),
            Conv2d(128, 32, kernel_size=3, padding=1),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(2, 2),
        )
        self.layer2 = Sequential(
            Linear(32 * 3 * 3, 128),
            ReLU(),
            Linear(128, 10),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.layer1(x)
        x = x.reshape((batch_size, -1))
        x = self.layer2(x)
        return x


if __name__ == "__main__":
    train_dataset = MNISTDataset("./datasets", True)
    test_dataset = MNISTDataset("./datasets", False)
    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = CNN()
    model.to("cuda:0")
    optimizer = SGD(model.parameters(), lr=1e-2)

    for epoch in range(3):
        model.train()
        for i, (x, y) in enumerate(train_data_loader):
            input_cpu_array = (
                ((x.cpu_array.reshape((-1, 1, 28, 28)) / 255.0) - 0.1307) / 0.3081
            ).astype(np.float32)
            input_tensor = Tensor(cpu_array=input_cpu_array, device="cuda:0")
            logits = model(input_tensor)
            loss = cross_entropy(logits, y.to("cuda:0"))
            optimizer.zero_grad()
            loss.backward()
            if (i + 1) % 16 == 0:
                loss = loss.item()
                accuracy = (
                    (logits.to("cpu").cpu_array.argmax(1) == y.cpu_array)
                    .astype(np.float32)
                    .mean()
                )
                print(
                    f"Epoch #{epoch}, step #{i}, accuracy: {accuracy* 100:0.2f}%, loss: {loss:0.4f}"
                )
            optimizer.step()

        model.eval()
        correct = 0
        for i, (x, y) in enumerate(test_data_loader):
            input_cpu_array = (
                ((x.cpu_array.reshape((-1, 1, 28, 28)) / 255.0) - 0.1307) / 0.3081
            ).astype(np.float32)
            input_tensor = Tensor(cpu_array=input_cpu_array, device="cuda:0")
            with no_grad():
                logits = model(input_tensor)
            correct += (
                (logits.to("cpu").cpu_array.argmax(1) == y.cpu_array)
                .astype(np.float32)
                .sum()
                .item()
            )
        accuracy = correct / len(test_dataset)
        print(f"Epoch #{epoch}, test accuracy: {accuracy *100:0.2f}%")
