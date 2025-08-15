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
