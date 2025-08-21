from mytorch.tensor import Tensor


class Parameter(Tensor):
    def __init__(self, tensor: Tensor):
        super().__init__(tensor=tensor, requires_grad=True)
