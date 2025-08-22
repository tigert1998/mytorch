import numpy as np
from typing import Optional

from cuda.bindings import driver

from mytorch.cuda.env import check_cuda_errors, CudaEnv
from mytorch.autograd import DAGTracker


class InvalidDeviceError(RuntimeError):
    def __init__(self, device_type):
        message = f"Invalid device type: {device_type}"
        super().__init__(message)


class InvalidDataTypeError(RuntimeError):
    def __init__(self, data_type):
        message = f"Invalid data type: {data_type}"
        super().__init__(message)


def shape_size(shape):
    from functools import reduce

    return reduce(lambda x, y: x * y, shape, 1)


class Device:
    type: str
    index: Optional[int]

    def __init__(self, device):
        if isinstance(device, Device):
            self.type = device.type
            self.index = device.index
        else:
            device = device.strip()
            if device in ["cpu"]:
                self.type = device
                self.index = None
            else:
                sep = device.find(":")
                self.type = device[:sep].strip()
                self.index = int(device[sep + 1 :])

    def __eq__(self, other):
        return (
            isinstance(other, Device)
            and self.type == other.type
            and self.index == other.index
        )

    def __repr__(self):
        if self.index is None:
            return self.type
        else:
            return f"{self.type}:{self.index}"


class CudaMemory:
    def __init__(self, size):
        self.ptr = CudaEnv.instance().allocator.allocate(size)

    def destroy(self):
        if self.ptr is not None:
            CudaEnv.instance().allocator.deallocate(self.ptr)
        self.ptr = None

    def __del__(self):
        self.destroy()

    def read(self, shape, dtype) -> np.ndarray:
        array = np.zeros(shape=shape, dtype=dtype)
        check_cuda_errors(
            driver.cuMemcpyDtoH(
                array.ctypes.data,
                self.ptr,
                array.itemsize * array.size,
            )
        )
        return array

    def write(self, array: np.ndarray):
        check_cuda_errors(
            driver.cuMemcpyHtoD(
                self.ptr,
                array.ctypes.data,
                array.itemsize * array.size,
            )
        )


class Tensor:
    def __init__(
        self,
        cpu_array: np.ndarray = None,
        dtype=None,
        shape=None,
        device="cpu",
        tensor=None,
        requires_grad=False,
    ):
        self.requires_grad = requires_grad
        self.grad = None

        if tensor is not None:
            self.cpu_array = tensor.cpu_array
            self.cuda_ptr = tensor.cuda_ptr
            self.dtype = tensor.dtype
            self.shape = tensor.shape
            self.device = tensor.device
            return

        self.device: Device = Device(device)
        if self.device.type not in [
            "cpu",
            "cuda",
        ]:
            raise InvalidDeviceError(self.device.type)

        self.dtype = dtype
        self.shape = shape
        self.cuda_ptr = None
        self.cpu_array = None

        cuda_context_manager = CudaEnv.instance().context_manager

        if cpu_array is not None:
            assert self.dtype is None or self.dtype == cpu_array.dtype
            self.dtype = cpu_array.dtype
            assert self.shape is None or self.shape == cpu_array.shape
            self.shape = cpu_array.shape

            if self.device.type == "cpu":
                self.cpu_array = cpu_array
            elif self.device.type == "cuda":
                # copy from cpu to cuda
                cuda_context_manager.set_device(self.device.index)
                self.cuda_ptr = CudaMemory(cpu_array.itemsize * cpu_array.size)
                self._write_cuda_memory(cpu_array)

        elif self.device.type == "cpu":
            self.cpu_array = np.zeros(shape=self.shape, dtype=self.dtype)

        elif self.device.type == "cuda":
            cuda_context_manager.set_device(self.device.index)
            self.cuda_ptr = CudaMemory(
                np.dtype(self.dtype).itemsize * shape_size(self.shape)
            )

    def to(self, device):
        device = Device(device)
        if self.device == device:
            return self

        if self.device.type == "cpu":
            # cpu to cuda
            return Tensor(cpu_array=self.cpu_array, device=device)
        elif self.device.type == "cuda":
            if device.type == "cpu":
                # cuda to cpu
                array = self._read_cuda_memory()
                return Tensor(cpu_array=array, device="cpu")
            elif device.type == "cuda:":
                # cuda to cuda
                new_tensor = Tensor(dtype=self.dtype, shape=self.shape, device=device)
                check_cuda_errors(
                    driver.cuMemcpyPeer(
                        new_tensor.cuda_ptr.ptr,
                        check_cuda_errors(driver.cuDeviceGet(new_tensor.device.index)),
                        self.cuda_ptr.ptr,
                        check_cuda_errors(driver.cuDeviceGet(self.device.index)),
                        np.dtype(self.dtype).itemsize * shape_size(self.shape),
                    )
                )
                return new_tensor

    def _read_cuda_memory(self) -> np.ndarray:
        return self.cuda_ptr.read(self.shape, self.dtype)

    def _write_cuda_memory(self, array: np.ndarray):
        self.cuda_ptr.write(array)

    def __repr__(self):
        if self.device.type == "cpu":
            array = self.cpu_array
        elif self.device.type == "cuda":
            array = self._read_cuda_memory()
        return f'tensor({repr(array)}, device="{self.device}")'

    def fill_(self, value):
        from mytorch.elementwise_ops import _fill

        _fill(self, value)

    def copy_(self, tensor):
        from mytorch.basic_ops import _copy

        assert isinstance(tensor, Tensor) and self.dtype == tensor.dtype
        _copy(self, tensor.to(self.device))

    def sum(self, dim=None, keepdim=False):
        from mytorch.basic_ops import sum as func

        return func(self, dim, keepdim)

    def _cast_other_to_tensor(self, other):
        import numbers

        if isinstance(other, numbers.Number):
            other = Tensor(
                cpu_array=np.array(other, dtype=self.dtype), device=self.device
            )
        elif isinstance(other, np.ndarray):
            other = Tensor(cpu_array=other.astype(self.dtype), device=self.device)
        elif isinstance(other, Tensor):
            pass
        else:
            raise InvalidDataTypeError(type(other))

        return other

    def __add__(self, other):
        from mytorch.basic_ops import add

        return add(self, self._cast_other_to_tensor(other))

    def __radd__(self, other):
        from mytorch.basic_ops import add

        return add(self._cast_other_to_tensor(other), self)

    def __sub__(self, other):
        from mytorch.basic_ops import sub

        return sub(self, self._cast_other_to_tensor(other))

    def __rsub__(self, other):
        from mytorch.basic_ops import sub

        return sub(self._cast_other_to_tensor(other), self)

    def __mul__(self, other):
        from mytorch.basic_ops import mul

        return mul(self, self._cast_other_to_tensor(other))

    def __rmul__(self, other):
        from mytorch.basic_ops import mul

        return mul(self._cast_other_to_tensor(other), self)

    def __truediv__(self, other):
        from mytorch.basic_ops import div

        return div(self, self._cast_other_to_tensor(other))

    def __rtruediv__(self, other):
        from mytorch.basic_ops import div

        return div(self._cast_other_to_tensor(other), self)

    def __pow__(self, other):
        from mytorch.basic_ops import pow

        return pow(self, self._cast_other_to_tensor(other))

    def __rpow__(self, other):
        from mytorch.basic_ops import pow

        return pow(self._cast_other_to_tensor(other), self)

    def backward(self):
        instance = DAGTracker.instance()
        instance.backward(self)
