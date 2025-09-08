from mytorch.backends.backend_dispatcher import BackendDispatcher
from mytorch.autograd import DAGTracker


def max_pool2d(x, kernel_size, stride=None, padding=0):
    func = BackendDispatcher.instance().dispatch(x.device.type, "max_pool2d")
    tensor = func(x, kernel_size, stride, padding)
    tensor.requires_grad = x.requires_grad
    if tensor.requires_grad:
        DAGTracker.instance().add_node("max_pool2d", [x, kernel_size, stride, padding], [tensor])
    return tensor


@DAGTracker.instance().register_backward_function("max_pool2d")
def max_pool2d_backward(output_grad, output, input, kernel_size, stride, padding):
    func = BackendDispatcher.instance().dispatch(input.device.type, "max_pool2d_backward")
    return func(output_grad, output, input, kernel_size, stride, padding)
