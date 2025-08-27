import numpy as np

from mytorch.tensor import Tensor, InvalidDeviceError
from mytorch.autograd import DAGTracker


def cat(tensors, dim):
    requires_grad = tensors[0].requires_grad
    dtype = tensors[0].dtype
    device = tensors[0].device
    shape = list(tensors[0].shape)
    for i in range(1, len(tensors)):
        assert requires_grad == tensors[i].requires_grad
        assert dtype == tensors[i].dtype
        assert device == tensors[i].device
        assert len(shape) == len(tensors[i].shape)
        for j in range(len(shape)):
            assert j == dim or shape[j] == tensors[i].shape[j]
        shape[dim] += tensors[i].shape[dim]

    output_tensor = Tensor(
        dtype=dtype,
        shape=shape,
        device=device,
        requires_grad=requires_grad,
    )

    if device.type == "cuda":
        raise NotImplementedError()

    elif device.type == "cpu":
        cpu_arrays = [tensor.cpu_array for tensor in tensors]
        output_tensor.cpu_array = np.concatenate(cpu_arrays, axis=dim)

    else:
        raise InvalidDeviceError(device.type)

    if requires_grad:
        DAGTracker.instance().add_node("cat", [*tensors, dim], [output_tensor])

    return output_tensor


@DAGTracker.instance().register_backward_function("cat")
def cat_backward(output_grad, *args):
    tensors = args[:-1]
    dim = args[-1]
    if output_grad.device.type == "cuda":
        raise NotImplementedError()
    elif output_grad.device.type == "cpu":
        last = 0
        splits = []
        for tensor in tensors:
            last += tensor.shape[dim]
            splits.append(last)
        splits = splits[:-1]
        arrays = np.split(output_grad.cpu_array, splits, axis=dim)
        tensors_grads = [Tensor(cpu_array=array) for array in arrays]
    else:
        raise InvalidDeviceError(output_grad.device.type)
    return tensors_grads
