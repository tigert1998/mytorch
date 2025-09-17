from mytorch.optim.optimizer import Optimizer
from mytorch.tensor import Tensor
from mytorch.backends.backend_dispatcher import BackendDispatcher


class SGD(Optimizer):
    def __init__(
        self,
        params,
        lr=0.001,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        maximize=False,
    ):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "maximize": maximize,
        }

        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for param in group["params"]:
                if param.shape != param.grad.shape:
                    raise RuntimeError(
                        f"Mismatched shape between param and param.grad: {param}, {param.grad}"
                    )

                self.state.setdefault(param, {})
                is_first_time = self.state[param].get("momentum_buffer") is None
                self.state[param].setdefault(
                    "momentum_buffer",
                    Tensor(shape=param.shape, device=param.device, dtype=param.dtype),
                )

                momentum_buffer = self.state[param]["momentum_buffer"]

                func = BackendDispatcher.instance().dispatch(param.device.type, "sgd")
                func(
                    param,
                    momentum_buffer,
                    is_first_time,
                    group["lr"],
                    group["weight_decay"],
                    group["momentum"],
                    group["dampening"],
                    group["nesterov"],
                    group["maximize"],
                )
