import numpy as np
from typing import Optional

from cuda.bindings import runtime

from cuda_utils import check_cuda_errors


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
        check_cuda_errors(runtime.cudaFree(self.ptr))


class Tensor:
    def __init__(
        self,
        cpu_array: np.ndarray = None,
        dtype=None,
        shape=None,
        device="cpu",
        tensor=None,
    ):
        if tensor is not None:
            self.cpu_array = tensor.cpu_array
            self.cuda_ptr = tensor.cuda_ptr
            self.dtype = tensor.dtype
            self.shape = tensor.shape
            self.device = tensor.device
            return

        self.device = Device(device)
        assert self.device.type in [
            "cpu",
            "cuda",
        ], f"Invalid device type: {self.device.type}"

        self.dtype = dtype
        self.shape = shape
        self.cuda_ptr = None
        self.cpu_array = None

        if cpu_array is not None:
            assert self.dtype is None or self.dtype == cpu_array.dtype
            self.dtype = cpu_array.dtype
            assert self.shape is None or self.shape == cpu_array.shape
            self.shape = cpu_array.shape

            if self.device.type == "cpu":
                self.cpu_array = cpu_array
            elif self.device.type == "cuda":
                # copy from cpu to cuda
                check_cuda_errors(runtime.cudaSetDevice(self.device.index))
                self.cuda_ptr = CudaPtr(
                    check_cuda_errors(
                        runtime.cudaMalloc(cpu_array.itemsize * cpu_array.size)
                    )
                )
                check_cuda_errors(
                    runtime.cudaMemcpy(
                        self.cuda_ptr.ptr,
                        cpu_array.ctypes.data,
                        cpu_array.itemsize * cpu_array.size,
                        runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    )
                )

        elif self.device.type == "cpu":
            self.cpu_array = np.zeros(shape=self.shape, dtype=self.dtype)

        elif self.device.type == "cuda":
            check_cuda_errors(runtime.cudaSetDevice(self.device.index))
            self.cuda_ptr = CudaPtr(
                check_cuda_errors(
                    runtime.cudaMalloc(
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
                array = np.zeros(shape=self.shape, dtype=self.dtype)
                check_cuda_errors(
                    runtime.cudaMemcpy(
                        array.ctypes.data,
                        self.cuda_ptr.ptr,
                        array.itemsize * array.size,
                        runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    )
                )
                return Tensor(cpu_array=array, device="cpu")
            elif device.type == "cuda:":
                # cuda to cuda
                new_tensor = Tensor(dtype=self.dtype, shape=self.shape, device=device)
                check_cuda_errors(
                    runtime.cudaMemcpy(
                        new_tensor.cuda_ptr.ptr,
                        self.cuda_ptr.ptr,
                        np.dtype(self.dtype).itemsize * np.prod(self.shape),
                        runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                    )
                )
                return new_tensor
