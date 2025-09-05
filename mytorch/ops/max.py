from mytorch.autograd import DAGTracker
from mytorch.backends.backend_dispatcher import BackendDispatcher


def max(tensor, dim=None, keepdim=False):
    func = BackendDispatcher.instance().dispatch(tensor.device.type, "max")
    tensors = func(tensor, dim, keepdim)
    output_tensor, indices_tensor = tensors
    output_tensor.requires_grad = tensor.requires_grad
    if output_tensor.requires_grad:
        DAGTracker.instance().add_node(
            "max", [tensor, dim, keepdim], [output_tensor], [indices_tensor]
        )
    if dim is not None:
        return tensors
    else:
        return output_tensor


@DAGTracker.instance().register_backward_function("max")
def max_backward(output_grad, indices_tensor, tensor, dim, keepdim):
    func = BackendDispatcher.instance().dispatch(tensor.device.type, "max_backward")
    return func(output_grad, indices_tensor, tensor, dim, keepdim)
