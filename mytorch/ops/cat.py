from mytorch.autograd import DAGTracker
from mytorch.backends.backend_dispatcher import BackendDispatcher


def cat(tensors, dim):
    func = BackendDispatcher.instance().dispatch(tensors[0].device.type, "cat")
    output_tensor = func(tensors, dim)
    output_tensor.requires_grad = (
        tensors[0].requires_grad and not DAGTracker.instance().no_grad
    )
    if output_tensor.requires_grad:
        DAGTracker.instance().add_node("cat", [*tensors, dim], [output_tensor])
    return output_tensor


@DAGTracker.instance().register_backward_function("cat")
def cat_backward(output_grad, *args):
    func = BackendDispatcher.instance().dispatch(
        output_grad.device.type, "cat_backward"
    )
    tensors_grads = func(output_grad, *args)
    return tensors_grads
