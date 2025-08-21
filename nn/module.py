from typing import Optional

from nn.parameter import Parameter


class Module:
    def __init__(self):
        self._parameters = {}

    def register_parameter(self, name, parameter: Optional[Parameter]):
        if parameter is not None:
            self._parameters[name] = parameter
        self.__setattr__(name, parameter)

    def to(self, device):
        for key in self._parameters.keys():
            self._parameters[key] = self._parameters[key].to(device)

    def __getattr__(self, name):
        if name in self._parameters:
            return self._parameters[name]
        return getattr(self, name)

    def parameters(self):
        for param in self._parameters.values():
            yield param

    def named_parameters(self):
        for name, param in self._parameters.values():
            yield (name, param)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
