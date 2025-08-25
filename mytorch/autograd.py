import numpy as np


class DAGTracker:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = DAGTracker()
        return cls._instance

    def register_backward_function(self, op_name):
        def decorator(func):
            self._backward_funcs[op_name] = func
            return func

        return decorator

    def no_grad(self, value):
        self._no_grad = value

    def __init__(self):
        self._reset_dag()
        self._backward_funcs = {}
        self._no_grad = False

    def _reset_dag(self):
        self._tensor_from_node = {}
        self._tensor_to_nodes = {}
        self._node_input_args = {}
        self._node_outputs = {}
        self._node_type_counts = {}

    def add_node(self, name, input_args, output_tensors):
        if self._no_grad:
            return

        from mytorch.tensor import Tensor

        index = self._node_type_counts.get(name, 0)
        self._node_type_counts[name] = index + 1
        node = (name, index)

        for output_tensor in output_tensors:
            self._tensor_from_node[output_tensor] = node

        for input_tensor in input_args:
            if isinstance(input_tensor, Tensor):
                self._tensor_to_nodes[input_tensor] = self._tensor_to_nodes.get(
                    input_tensor, []
                ) + [node]

        self._node_input_args[node] = input_args
        self._node_outputs[node] = output_tensors

    def _dfs(self, tensor, node, memo: set):
        from mytorch.tensor import Tensor

        if tensor is not None:
            if tensor in memo:
                return
            memo.add(tensor)
            self._dfs(None, self._tensor_from_node.get(tensor), memo)
            for next_node in self._tensor_to_nodes.get(tensor, []):
                self._dfs(None, next_node, memo)
        elif node is not None:
            if node in memo:
                return
            memo.add(node)
            for next_tensor in self._node_input_args[node]:
                if isinstance(next_tensor, Tensor):
                    self._dfs(next_tensor, None, memo)
            for next_tensor in self._node_outputs[node]:
                self._dfs(next_tensor, None, memo)

    def _topological_sort(self, tensor):
        from mytorch.tensor import Tensor

        memo = set()
        self._dfs(tensor, None, memo)

        outdegree = {}
        for node in self._tensor_from_node.values():
            outdegree[node] = outdegree.get(node, 0) + 1
        for input_args in self._node_input_args.values():
            for input_tensor in input_args:
                if isinstance(input_tensor, Tensor):
                    outdegree[input_tensor] = outdegree.get(input_tensor, 0) + 1

        order = []

        tensors_to_remove = initial_tensors = [
            tensor
            for tensor in memo
            if outdegree.get(tensor, 0) == 0 and isinstance(tensor, Tensor)
        ]
        nodes_to_remove = []
        while len(tensors_to_remove) + len(nodes_to_remove) >= 1:
            for tensor_to_remove in tensors_to_remove:
                node = self._tensor_from_node.get(tensor_to_remove)
                if node is None:
                    # tensor_to_remove is leaf tensor
                    continue
                outdegree[node] -= 1
                if outdegree[node] == 0:
                    nodes_to_remove.append(node)
                    order.append(node)
            tensors_to_remove = []

            for node_to_remove in nodes_to_remove:
                for input_tensor in self._node_input_args[node_to_remove]:
                    if isinstance(input_tensor, Tensor):
                        outdegree[input_tensor] -= 1
                        if outdegree[input_tensor] == 0:
                            tensors_to_remove.append(input_tensor)
            nodes_to_remove = []

        return order, initial_tensors

    def backward(self, tensor):
        from mytorch.tensor import Tensor, shape_size

        order, initial_tensors = self._topological_sort(tensor)
        assert len(order) >= 1, "make sure you don't backward a tensor twice"
        assert tensor in initial_tensors
        assert shape_size(tensor.shape) == 1
        tensor.grad = Tensor(
            cpu_array=np.array(1, dtype=tensor.dtype).reshape(tensor.shape),
            device=tensor.device,
        )
        for initial_tensor in initial_tensors:
            if initial_tensor is tensor:
                continue
            initial_tensor.grad = Tensor(
                shape=initial_tensor.shape,
                device=initial_tensor.device,
                dtype=initial_tensor.dtype,
            )
            initial_tensor.grad.fill_(0)

        for node, idx in order:
            backward_func = self._backward_funcs[node]
            output_tensors = self._node_outputs[(node, idx)]
            output_tensor_grads = [tensor.grad for tensor in output_tensors]
            input_tensors_grads = backward_func(
                *output_tensor_grads, *self._node_input_args[(node, idx)]
            )
            for input_tensor, grad in zip(
                self._node_input_args[(node, idx)], input_tensors_grads
            ):
                if input_tensor.requires_grad:
                    if input_tensor.grad is None:
                        input_tensor.grad = grad
                    else:
                        assert input_tensor.grad.shape == grad.shape
                        input_tensor.grad += grad

        # erase tensor and nodes from the connected dag
        for node, idx in order:
            for output_tensor in self._node_outputs[(node, idx)]:
                del self._tensor_from_node[output_tensor]
            for input_tensor in self._node_input_args[(node, idx)]:
                if (
                    isinstance(input_tensor, Tensor)
                    and input_tensor in self._tensor_to_nodes
                ):
                    del self._tensor_to_nodes[input_tensor]

        for node, idx in order:
            del self._node_outputs[(node, idx)]
            del self._node_input_args[(node, idx)]


class no_grad:
    def __enter__(self):
        DAGTracker.instance().no_grad(True)

    def __exit__(self, exc_type, exc_value, traceback):
        DAGTracker.instance().no_grad(False)
