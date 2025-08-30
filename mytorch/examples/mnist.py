import numpy as np
import argparse

import mytorch
import mytorch.nn as nn
import mytorch.optim as optim
import mytorch.nn.functional as F
from mytorch.vision.datasets import MNIST
from mytorch.utils.data import DataLoader
import mytorch.vision.transforms as transforms


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(32 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.layer1(x)
        x = x.reshape((batch_size, -1))
        x = self.layer2(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train MNIST with MyTorch")
    parser.add_argument("--ckpt")
    parser.add_argument("--save-ckpt")
    args = parser.parse_args()

    transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)]
    )

    train_dataset = MNIST("./datasets", True, transform=transforms)
    test_dataset = MNIST("./datasets", False, transform=transforms)
    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = CNN()
    model.to("cuda:0")
    optimizer = optim.SGD(model.parameters(), lr=1e-1)

    last_epoch = -1
    if args.ckpt is not None:
        ckpt = mytorch.load(args.ckpt, map_location="cuda:0")
        optimizer.load_state_dict(ckpt["optimizer"])
        model.load_state_dict(ckpt["model"])
        last_epoch = ckpt["epoch"]
        accuracy = ckpt["accuracy"] * 100
        print(f"Accuracy: {accuracy:.2f}%")

    for epoch in range(last_epoch + 1, last_epoch + 2):
        model.train()
        for i, (x, y) in enumerate(train_data_loader):
            input_tensor = x.to("cuda:0")
            target = y.to("cuda:0", np.int64)
            logits = model(input_tensor)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            if (i + 1) % 16 == 0:
                loss = loss.item()
                accuracy = (
                    logits.max(dim=1)[1].eq(target).to(dtype=np.float32).mean()
                ).item()
                print(
                    f"Epoch #{epoch}, step #{i}, accuracy: {accuracy* 100:0.2f}%, loss: {loss:0.4f}"
                )
            optimizer.step()

        model.eval()
        correct = 0
        for i, (x, y) in enumerate(test_data_loader):
            input_tensor = x.to("cuda:0")
            target = y.to("cuda:0", np.int64)
            with mytorch.no_grad():
                logits = model(input_tensor)
            correct += logits.max(dim=1)[1].eq(target).to(dtype=np.float32).sum().item()
        accuracy = correct / len(test_dataset)
        print(f"Epoch #{epoch}, test accuracy: {accuracy *100:0.2f}%")

        if args.save_ckpt:
            mytorch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "model": model.state_dict(),
                    "accuracy": accuracy,
                    "epoch": epoch,
                },
                args.save_ckpt,
            )
