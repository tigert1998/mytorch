import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from mytorch.nn.linear import Linear
from mytorch.optim.sgd import SGD
from mytorch.tensor import Tensor


def make_data(batch_size):
    k = (np.random.rand(1) * 1.9 - 0.95).item()
    x = np.random.rand(batch_size) * 10 - 5
    b = (np.random.rand(1) * 10 - 5).item()
    noise = np.random.rand(batch_size) - 0.5
    y = x * k + b + noise
    return x, y, k, b


def linear_regression(x, y, device):
    linear = Linear(1, 1, bias=True, device=device)
    optimizer = SGD(linear.parameters(), lr=1e-2)
    input_tensor = Tensor(
        cpu_array=x.astype(np.float32).reshape((-1, 1)), device=device
    )
    ans = Tensor(cpu_array=y.astype(np.float32).reshape((len(x), 1)), device=device)

    for i in tqdm(range(500)):
        optimizer.zero_grad()
        output_tensor = linear(input_tensor)
        minus = output_tensor - ans
        power2 = minus**2
        loss = power2.sum() / len(x)
        loss.backward()
        optimizer.step()

    k = linear.weight.to("cpu").cpu_array.item()
    b = linear.bias.to("cpu").cpu_array.item()
    return k, b


if __name__ == "__main__":
    x, y, k, b = make_data(256)
    test_k, test_b = linear_regression(x, y, "cuda:0")
    print(f"Answer: {k}, {b}\nOutput: {test_k}, {test_b}")

    plt.scatter(x, y)
    plt.plot(x, x * k + b, label="ground truth")
    plt.plot(x, x * test_k + test_b, label="gradient descent")
    plt.legend()
    plt.show()
