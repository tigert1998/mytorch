from mytorch.nn.module import Module


class Sequential(Module):
    def __init__(self, *args):
        super().__init__()

        for i, module in enumerate(args):
            self.register_module(str(i), module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input
