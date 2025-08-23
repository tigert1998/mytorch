from typing import Optional, Self, Dict

from mytorch.nn.parameter import Parameter, Tensor


class Module:
    def __init__(self):
        self._parameters: Dict[str, Parameter] = {}
        self._buffers: Dict[str, Tensor] = {}
        self._modules: Dict[str, Self] = {}

    def register_parameter(self, name, parameter: Optional[Parameter]):
        if parameter is not None:
            self._parameters[name] = parameter
        super().__setattr__(name, parameter)

    def register_buffer(self, name, buffer: Optional[Tensor]):
        if buffer is not None:
            self._buffers[name] = buffer
        super().__setattr__(name, buffer)

    def register_module(self, name, module: Optional[Self]):
        if module is not None:
            self._modules[name] = module
        super().__setattr__(name, module)

    def to(self, device):
        for key in self._parameters.keys():
            self._parameters[key] = self._parameters[key].to(device)

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self.register_module(name, value)
        elif isinstance(value, Parameter):
            self.register_parameter(name, value)
        elif isinstance(value, Tensor):
            self.register_buffer(name, value)
        else:
            super().__setattr__(name, value)

    def parameters(self, recurse=True):
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def buffers(self, recurse=True):
        for _, buffer in self.named_buffers(recurse=recurse):
            yield buffer

    def named_parameters(self, recurse=True):
        for name, param in self._parameters.items():
            yield (name, param)
        for module in self._modules.values():
            yield from module.named_parameters(recurse=recurse)

    def named_buffers(self, recurse=True):
        for name, buffer in self._buffers.items():
            yield (name, buffer)
        for module in self._modules.values():
            yield from module.named_buffers(recurse=recurse)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
