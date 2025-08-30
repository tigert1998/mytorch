from typing import Optional, Self, Dict, Set

from mytorch.nn.parameter import Parameter, Tensor


class Module:
    def __init__(self):
        self._parameters: Dict[str, Parameter] = {}
        self._buffers: Dict[str, Tensor] = {}
        self._modules: Dict[str, Self] = {}
        self.training: bool = True

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

    def train(self, mode=True) -> Self:
        self.training = mode
        for child in self.children():
            child.train(mode=mode)
        return self

    def eval(self) -> Self:
        return self.train(mode=False)

    def to(self, device):
        for key in self._parameters.keys():
            self.register_parameter(key, self._parameters[key].to(device))
        for key in self._buffers.keys():
            self.register_buffer(key, self._buffers[key].to(device))
        for module in self._modules.values():
            module.to(device)

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

    def children(self):
        for _, module in self.named_children():
            yield module

    def _named_members(
        self, get_members_fn, prefix="", recurse=True, remove_duplicate: bool = True
    ):
        # from pytorch source code
        memo = set()
        modules = (
            self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate)
            if recurse
            else [(prefix, self)]
        )
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                if remove_duplicate:
                    memo.add(v)
                name = module_prefix + ("." if module_prefix else "") + k
                yield name, v

    def named_modules(
        self,
        memo: Optional[Set[Self]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        # from pytorch source code
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                yield from module.named_modules(
                    memo, submodule_prefix, remove_duplicate
                )

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ):
        # from pytorch source code
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ):
        # from pytorch source code
        gen = self._named_members(
            lambda module: module._buffers.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    def named_children(self):
        # from pytorch source code
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.detach()
        for name, buf in self._buffers.items():
            if buf is not None:
                destination[prefix + name] = buf if keep_vars else buf.detach()

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(
                    destination=destination,
                    prefix=prefix + name + ".",
                    keep_vars=keep_vars,
                )
        return destination

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
