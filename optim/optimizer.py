from typing import List, Dict, Any

from tensor import Tensor


class Optimizer:
    def __init__(self, params, defaults):
        self.param_groups: List[Dict[str, Any]] = []
        self.defaults = defaults
        self.state = {}

        param_groups = list(params)
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]

        for param_group in param_groups:
            self._add_param_group(param_group)

    def _add_param_group(self, param_group):
        param_group["params"] = list(param_group["params"])
        for key in self.defaults.keys():
            param_group.setdefault(key, self.defaults[key])

        for param in param_group["params"]:
            assert isinstance(param, Tensor)
        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group["params"]))
        assert param_set.isdisjoint(set(param_group["params"]))

        self.param_groups.append(param_group)

    def zero_grad(self):
        for g in self.param_groups:
            for param in g["params"]:
                param.grad = None
