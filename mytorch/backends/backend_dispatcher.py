from typing import Callable
from collections import defaultdict


class BackendDispatcher:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = BackendDispatcher()
            cls._instance._register_modules()
        return cls._instance

    def __init__(self):
        self._backend_impls = defaultdict(dict)

    def _register_modules(self):
        import mytorch.backends.cpu
        import mytorch.backends.cuda

    def register_backend_function(self, backend: str, op_name: str):
        def decorator(func):
            if self._backend_impls[backend].get(op_name) is not None:
                raise RuntimeError(
                    f"There's already a function registered for {op_name} on {backend} backend"
                )
            self._backend_impls[backend][op_name] = func

            return func

        return decorator

    def dispatch(self, backend: str, op_name: str) -> Callable:
        if self._backend_impls[backend].get(op_name) is None:
            raise RuntimeError(
                f"There's no function registered for {op_name} on {backend} backend"
            )
        return self._backend_impls[backend][op_name]
