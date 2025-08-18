class DAGTracker:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = DAGTracker()
        return cls._instance

    def __init__(self):
        self._tensor_node = {}
        self._node_inputs = {}
        self._node_type_counts = {}

    def add_node(self, name, input_args, output_tensors):
        index = self._node_type_counts.get(name, 0)
        self._node_type_counts[name] = index + 1
        node = (name, index)

        for output_tensor in output_tensors:
            self._tensor_node[output_tensor] = node

        self._node_inputs[node] = input_args

    def _topological_sort(self, tensor):
        from tensor import Tensor

        ref_counts = {}
        for node in self._tensor_node.values():
            ref_counts[node] = ref_counts.get(node, 0) + 1
        for input_args in self._node_inputs.values():
            for input_tensor in input_args:
                if isinstance(input_tensor, Tensor):
                    ref_counts[input_tensor] = ref_counts.get(input_tensor, 0) + 1

        ans = []

        tensors_to_remove = [tensor]
        nodes_to_remove = []
        while len(tensors_to_remove) + len(nodes_to_remove) >= 1:
            for tensor_to_remove in tensors_to_remove:
                node = self._tensor_node.get(tensor_to_remove)
                if node is None:
                    # tensor_to_remove is leaf tensor
                    continue
                ref_counts[node] -= 1
                if ref_counts[node] == 0:
                    nodes_to_remove.append(node)
                    ans.append(node)
            tensors_to_remove = []

            for node_to_remove in nodes_to_remove:
                for input_tensor in self._node_inputs[node_to_remove]:
                    if isinstance(input_tensor, Tensor):
                        ref_counts[input_tensor] -= 1
                        if ref_counts[input_tensor] == 0:
                            tensors_to_remove.append(input_tensor)
            nodes_to_remove = []

        return ans

    def backward(self, tensor):
        order = self._topological_sort(tensor)
        print(order)
