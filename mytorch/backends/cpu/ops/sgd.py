from mytorch.tensor import Tensor
from mytorch.backends.backend_dispatcher import BackendDispatcher


@BackendDispatcher.instance().register_backend_function("cpu", "sgd")
def cpu_sgd(param: Tensor, momentum_buffer: Tensor, is_first_time: bool, lr, weight_decay, momentum, dampening,
            nesterov: bool, maximize: bool):
    g = (
        -param.grad._numpy()
        if maximize
        else param.grad._numpy()
    )
    g += weight_decay * param._numpy()
    momentum_buffer._cpu_array = (
        g
        if is_first_time
        else (
                momentum * momentum_buffer._numpy()
                + (1 - dampening) * g
        )
    )
    g = (
        (g + momentum * momentum_buffer._numpy())
        if nesterov
        else momentum_buffer._numpy()
    )
    param._cpu_array -= g * lr
