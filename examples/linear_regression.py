import numpy as np

from nn.linear import Linear
from optim.sgd import SGD
from tensor import Tensor


def make_data():
    k = (np.random.rand(1) * 1.9 - 0.95).item()
    x = np.random.rand(256) * 10 - 5
    b = (np.random.rand(1) * 10 - 5).item()
    noise = np.random.rand(256) * 1
    y = x * k + b + noise
    return x, y, k, b


def linear_regression(x, y, device):
    linear = Linear(1, 1, bias=True, device=device)
    optimizer = SGD(linear.parameters(), lr=1e-6)
    input_tensor = Tensor(
        cpu_array=x.astype(np.float32).reshape((-1, 1)), device=device
    )
    ans = Tensor(cpu_array=y.astype(np.float32), device=device)
    _2 = Tensor(cpu_array=np.array(2, dtype=np.float32), device=device)

    for i in range(100):
        optimizer.zero_grad()
        output_tensor = linear(input_tensor)
        loss = ((output_tensor - ans) ** _2).sum()
        loss.backward()
        print(linear.weight)
        print(linear.bias)
        optimizer.step()

    k = linear.weight.to("cpu").cpu_array.item()
    b = linear.bias.to("cpu").cpu_array.item()
    return k, b


if __name__ == "__main__":
    x, y, k, b = make_data()
    test_k, test_b = linear_regression(x, y, "cuda:0")
    print(f"Answer: {k}, {b}\nTest: {test_k}, {test_b}")
