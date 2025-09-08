import numpy as np

from mytorch.tensor import InvalidDataTypeError, InvalidDeviceError, Tensor
from mytorch.backends.backend_dispatcher import BackendDispatcher
from mytorch.dtype import int64


@BackendDispatcher.instance().register_backend_function("cpu", "cross_entropy")
def cpu_cross_entropy(input: Tensor, target: Tensor):
    if target.dtype != int64:
        raise InvalidDataTypeError(target.dtype)
    tensor = Tensor(
        dtype=input.dtype,
        shape=(1,),
        device=input.device,
    )
    softmax = np.exp(input._numpy()) / np.sum(
        np.exp(input._numpy()), axis=1, keepdims=True
    )
    log = [-np.log(softmax[i][target._numpy()[i]]) for i in range(target.shape[0])]
    tensor._cpu_array = np.array(np.mean(log), dtype=input.dtype.np_dtype)

    return tensor


@BackendDispatcher.instance().register_backend_function("cpu", "cross_entropy_backward")
def cpu_cross_entropy_backward(output_grad: Tensor, input: Tensor, target: Tensor):
    batch_size, num_classes = input.shape

    input_grad = Tensor(
        dtype=input.dtype,
        shape=input.shape,
        device=input.device,
    )

    softmax = np.exp(input._numpy()) / np.sum(
        np.exp(input._numpy()), axis=1, keepdims=True
    )
    target_onehot = np.eye(num_classes, dtype=input.dtype)[target._numpy()]
    input_grad._cpu_array = (
            (softmax - target_onehot) * output_grad._numpy() / batch_size
    )

    return [input_grad]
