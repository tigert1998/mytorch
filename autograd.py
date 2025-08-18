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

    def backward(self, tensor): ...
