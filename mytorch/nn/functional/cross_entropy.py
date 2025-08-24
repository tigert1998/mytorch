import numpy as np

from mytorch.tensor import InvalidDataTypeError, InvalidDeviceError, Tensor
from mytorch.cuda.env import CudaEnv
from mytorch.autograd import DAGTracker


def cross_entropy(input, target):
    batch_size, num_classes = input.shape

    requires_grad = input.requires_grad
    tensor = Tensor(
        dtype=input.dtype,
        shape=(1,),
        device=input.device,
        requires_grad=requires_grad,
    )

    if input.device.type == "cuda":
        if input.dtype == np.float32:
            func_name = "cross_entropy_reference_fp32"
        elif input.dtype == np.float16:
            func_name = "cross_entropy_reference_fp16"
        else:
            raise InvalidDataTypeError(input.dtype)
        cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
        cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
            "cross_entropy.cu", func_name, input.device.index
        )
        cuda_kernel.run(
            (32, 1, 1),
            (32, 1, 1),
            [
                np.array(batch_size),
                np.array(num_classes),
                input,
                target,
                tensor,
            ],
        )

    elif input.device.type == "cpu":
        softmax = np.exp(input.cpu_array) / np.sum(
            np.exp(input.cpu_array), axis=1, keepdims=True
        )
        log = [-np.log(softmax[i][target.cpu_array[i]]) for i in range(target.shape[0])]
        tensor.cpu_array = np.mean(log)

    else:
        raise InvalidDeviceError(input.device.type)

    if requires_grad:
        DAGTracker.instance().add_node("cross_entropy", [input, target], [tensor])

    return tensor


@DAGTracker.instance().register_backward_function("cross_entropy")
def cross_entropy_backward(output_grad, input, target):
    batch_size, num_classes = input.shape

    input_grad = Tensor(
        dtype=input.dtype,
        shape=input.shape,
        device=input.device,
    )

    if input.device.type == "cuda":
        if input.dtype == np.float32:
            func_name = "cross_entropy_backward_reference_fp32"
        elif input.dtype == np.float16:
            func_name = "cross_entropy_backward_reference_fp16"
        else:
            raise InvalidDataTypeError(input.dtype)
        cuda_kernel_and_stream_manager = CudaEnv.instance().kernel_and_stream_manager
        cuda_kernel = cuda_kernel_and_stream_manager.get_kernel(
            "cross_entropy.cu", func_name, input.device.index
        )
        cuda_kernel.run(
            (32, 1, 1),
            (32, 1, 1),
            [
                np.array(batch_size),
                np.array(num_classes),
                input,
                target,
                input_grad,
                output_grad,
            ],
        )

    elif input.device.type == "cpu":
        softmax = np.exp(input.cpu_array) / np.sum(
            np.exp(input.cpu_array), axis=1, keepdims=True
        )
        target_onehot = np.eye(num_classes, dtype=input.dtype)[target.cpu_array]
        input_grad.cpu_array = (
            (softmax - target_onehot) * output_grad.cpu_array / batch_size
        )

    else:
        raise InvalidDeviceError(input.device.type)

    return [input_grad]
