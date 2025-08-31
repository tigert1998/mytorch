from typing import List, Dict, Any

from mytorch.tensor import Tensor


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
            if not isinstance(param, Tensor):
                raise RuntimeError(f"There's a non-tensor value in params: {param}")
        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group["params"]))
        if not param_set.isdisjoint(set(param_group["params"])):
            raise RuntimeError("Param groups have overlap")

        self.param_groups.append(param_group)

    def zero_grad(self):
        for g in self.param_groups:
            for param in g["params"]:
                param.grad = None

    def state_dict(self):
        state = {}

        id_map = {}

        def convert_group(g):
            new_g = {k: g[k] for k in g.keys() if k != "params"}
            new_params = []
            for p in g["params"]:
                i = id_map.get(p)
                if i is None:
                    i = id_map[p] = id(p)
                new_params.append(i)
            new_g["params"] = new_params
            return new_g

        for k in self.state:
            state[id(k)] = self.state[k]

        param_groups = [convert_group(g) for g in self.param_groups]
        return {"state": state, "param_groups": param_groups}

    def load_state_dict(self, state_dict):
        state = state_dict["state"]
        param_groups = state_dict["param_groups"]

        if len(param_groups) != len(self.param_groups):
            raise RuntimeError(
                "Optimizer state_dict has a different param group length from the loaded one"
            )
        for i in range(len(param_groups)):
            if len(param_groups[i]["params"]) != len(self.param_groups[i]["params"]):
                raise RuntimeError(
                    "Optimizer state_dict is different from the loaded one"
                )

        for i in range(len(self.param_groups)):
            for j in range(len(self.param_groups[i]["params"])):
                idx = param_groups[i]["params"][j]
                if state.get(idx) is None:
                    continue
                tensor = self.param_groups[i]["params"][j]
                self.state[tensor] = state[idx]
