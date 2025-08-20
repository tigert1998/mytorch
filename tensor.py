import numpy as np
from typing import Optional

from cuda.bindings import driver

from cuda.cuda_utils import check_cuda_errors, CudaContextManager
from autograd import DAGTracker


class InvalidDeviceError(RuntimeError):
    def __init__(self, device_type):
        message = f"Invalid device type: {device_type}"
        super().__init__(message)


class InvalidDataTypeError(RuntimeError):
    def __init__(self, data_type):
        message = f"Invalid data type: {data_type}"
        super().__init__(message)


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


class CudaPtr:
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        check_cuda_errors(driver.cuMemFree(self.ptr))


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

        self.device = Device(device)
        if self.device.type not in [
            "cpu",
            "cuda",
        ]:
            raise InvalidDeviceError(self.device.type)

        self.dtype = dtype
        self.shape = shape
        self.cuda_ptr = None
        self.cpu_array = None

        cuda_context_manager = CudaContextManager.instance()

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
                self.cuda_ptr = CudaPtr(
                    check_cuda_errors(
                        driver.cuMemAlloc(cpu_array.itemsize * cpu_array.size)
                    )
                )
                self._write_cuda_memory(cpu_array)

        elif self.device.type == "cpu":
            self.cpu_array = np.zeros(shape=self.shape, dtype=self.dtype)

        elif self.device.type == "cuda":
            cuda_context_manager.set_device(self.device.index)
            self.cuda_ptr = CudaPtr(
                check_cuda_errors(
                    driver.cuMemAlloc(
                        np.dtype(self.dtype).itemsize * np.prod(self.shape)
                    )
                )
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
                        np.dtype(self.dtype).itemsize * np.prod(self.shape),
                    )
                )
                return new_tensor

    def _read_cuda_memory(self) -> np.ndarray:
        array = np.zeros(shape=self.shape, dtype=self.dtype)
        check_cuda_errors(
            driver.cuMemcpyDtoH(
                array.ctypes.data,
                self.cuda_ptr.ptr,
                array.itemsize * array.size,
            )
        )
        return array

    def _write_cuda_memory(self, array: np.ndarray):
        check_cuda_errors(
            driver.cuMemcpyHtoD(
                self.cuda_ptr.ptr,
                array.ctypes.data,
                array.itemsize * array.size,
            )
        )

    def __repr__(self):
        if self.device.type == "cpu":
            array = self.cpu_array
        elif self.device.type == "cuda":
            array = self._read_cuda_memory()
        return f'tensor({repr(array)}, device="{self.device}")'

    def fill_(self, value):
        if self.device.type == "cpu":
            self.cpu_array.fill(value)
        elif self.device.type == "cuda":
            # TODO: use kernel function
            array = self._read_cuda_memory()
            array.fill(value)
            self._write_cuda_memory(array)

    def copy_(self, tensor):
        assert self.dtype == tensor.dtype and self.shape == tensor.shape
        if self.device.type == "cpu":
            if tensor.device.type == "cpu":
                self.cpu_array = tensor.cpu_array
            elif tensor.device.type == "cuda":
                self.cpu_array = tensor._read_cuda_memory()
        elif self.device.type == "cuda":
            if tensor.device.type == "cpu":
                self._write_cuda_memory(tensor.cpu_array)
            elif tensor.device.type == "cuda":
                check_cuda_errors(
                    driver.cuMemcpyPeer(
                        self.cuda_ptr.ptr,
                        check_cuda_errors(driver.cuDeviceGet(self.device.index)),
                        tensor.cuda_ptr.ptr,
                        check_cuda_errors(driver.cuDeviceGet(tensor.device.index)),
                        np.dtype(self.dtype).itemsize * np.prod(self.shape),
                    )
                )

    def sum(self):
        from basic_ops import sum as func

        return func(self)

    def backward(self):
        instance = DAGTracker.instance()
        instance.backward(self)
