import pickle

from mytorch.tensor import Tensor


class MyTorchPickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, Tensor):
            return ("MyTorchTensor", obj.device, obj.to("cpu")._cpu_array)
        else:
            return None


class MyTorchUnpickler(pickle.Unpickler):
    def __init__(self, file, device=None):
        super().__init__(file)
        self.device = device

    def persistent_load(self, pid):
        if pid[0] == "MyTorchTensor":
            return Tensor(
                cpu_array=pid[2],
                device=self.device if self.device is not None else pid[1],
            )
        else:
            raise pickle.UnpicklingError("unsupported persistent object")


def save(obj, f):
    with open(f, "wb") as f:
        pickler = MyTorchPickler(f)
        pickler.dump(obj)


def load(f, map_location=None):
    with open(f, "rb") as f:
        unpickler = MyTorchUnpickler(f, map_location)
        return unpickler.load()
