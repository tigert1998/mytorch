import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple, List

from cuda.bindings import driver

from mytorch.dtype import DType
from mytorch.backends.cuda.env import check_cuda_errors, CudaEnv
from mytorch.autograd import DAGTracker


class InvalidDeviceError(RuntimeError):
    def __init__(self, device_type):
        message = f"Invalid device type: {device_type}"
        super().__init__(message)


class InvalidDataTypeError(RuntimeError):
    def __init__(self, data_type):
        message = f"Invalid data type: {data_type}"
        super().__init__(message)


class InvalidShapeError(RuntimeError):
    def __init__(self, shape):
        message = f"Invalid tensor shape: {shape}"
        super().__init__(message)


class MismatchDevicesError(RuntimeError):
    def __init__(self, device_types):
        message = f"Mismatch devices between tensors: {device_types}"
        super().__init__(message)


class MismatchDataTypesError(RuntimeError):
    def __init__(self, data_types):
        message = f"Mismatch data types between tensors: {data_types}"
        super().__init__(message)


class MismatchShapesError(RuntimeError):
    def __init__(self, shapes):
        message = f"Mismatch shapes between tensors: {shapes}"
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
            sep = device.find(":")
            if sep < 0:
                self.type = device
                self.index = None
            else:
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

    def read(self, shape, dtype: DType) -> npt.NDArray:
        array = np.zeros(shape=shape, dtype=dtype.np_dtype)
        check_cuda_errors(
            driver.cuMemcpyDtoH(
                array.ctypes.data,
                self.ptr,
                array.itemsize * array.size,
            )
        )
        return array

    def write(self, array: npt.NDArray):
        check_cuda_errors(
            driver.cuMemcpyHtoD(
                self.ptr,
                array.ctypes.data,
                array.itemsize * array.size,
            )
        )


def tensor(data, dtype=None, device=None, requires_grad=False):
    data = np.array(data)
    if dtype is not None:
        data = data.astype(dtype.np_dtype)
    return Tensor(
        cpu_array=data, dtype=dtype, device=device, requires_grad=requires_grad
    )


class Tensor:
    requires_grad: bool
    grad: Optional["Tensor"]
    cpu_array: Optional[npt.NDArray]
    dtype: DType
    shape: Tuple[int, ...]
    device: Device
    cuda_ptr: Optional[CudaMemory]

    def __init__(
        self,
        cpu_array: Optional[npt.NDArray] = None,
        dtype: Optional[DType] = None,
        shape: Optional[List[int] | Tuple[int, ...]] = None,
        device: Optional[str | Device] = None,
        tensor: Optional["Tensor"] = None,
        requires_grad: bool = False,
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

        # set device
        self.device: Device = Device(device if device is not None else "cpu")
        if self.device.type not in [
            "cpu",
            "cuda",
        ]:
            raise InvalidDeviceError(self.device.type)

        # set dtype
        if dtype is not None:
            if not isinstance(dtype, DType):
                raise InvalidDataTypeError(dtype)
            else:
                self.dtype = dtype
                if cpu_array is not None and self.dtype.np_dtype != cpu_array.dtype:
                    raise InvalidDataTypeError(self.dtype)
        elif cpu_array is not None:
            self.dtype = DType.from_np_dtype(cpu_array.dtype)

        # set shape
        if shape is not None:
            self.shape = tuple(shape)
            if cpu_array is not None and self.shape != cpu_array.shape:
                raise InvalidShapeError(self.shape)
        elif cpu_array is not None:
            self.shape = cpu_array.shape

        # set cpu/cuda memory
        self.cuda_ptr = None
        self.cpu_array = None

        cuda_context_manager = CudaEnv.instance().context_manager

        if cpu_array is not None:
            if self.device.type == "cpu":
                self.cpu_array = cpu_array
            elif self.device.type == "cuda":
                # copy from cpu to cuda
                cuda_context_manager.set_device(self.device.index)
                self.cuda_ptr = CudaMemory(cpu_array.itemsize * cpu_array.size)
                self._write_cuda_memory(cpu_array)

        elif self.device.type == "cpu":
            self.cpu_array = np.zeros(shape=self.shape, dtype=self.dtype.np_dtype)

        elif self.device.type == "cuda":
            cuda_context_manager.set_device(self.device.index)
            self.cuda_ptr = CudaMemory(self.dtype.itemsize() * shape_size(self.shape))

        if self.requires_grad and not self.dtype.is_floating:
            raise RuntimeError(f"tensor of type {self.dtype} cannot require gradient")

    def _to_device(self, device):
        device = Device(device)
        if self.device == device:
            return self

        if self.device.type == "cpu":
            # cpu to cuda
            return Tensor(
                cpu_array=self.cpu_array,
                device=device,
                requires_grad=self.requires_grad,
            )
        elif self.device.type == "cuda":
            if device.type == "cpu":
                # cuda to cpu
                array = self._read_cuda_memory()
                return Tensor(
                    cpu_array=array, device="cpu", requires_grad=self.requires_grad
                )
            elif device.type == "cuda:":
                # cuda to cuda
                new_tensor = Tensor(
                    dtype=self.dtype,
                    shape=self.shape,
                    device=device,
                    requires_grad=self.requires_grad,
                )
                check_cuda_errors(
                    driver.cuMemcpyPeer(
                        new_tensor._cuda_ptr(),
                        check_cuda_errors(driver.cuDeviceGet(new_tensor.device.index)),
                        self._cuda_ptr(),
                        check_cuda_errors(driver.cuDeviceGet(self.device.index)),
                        np.dtype(self.dtype).itemsize * shape_size(self.shape),
                    )
                )
                return new_tensor

    def _to_dtype(self, dtype):
        from mytorch.ops.cast import _cast

        return _cast(self, dtype)

    def to(self, device=None, dtype=None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        return self._to_device(device)._to_dtype(dtype)

    def _read_cuda_memory(self) -> npt.NDArray:
        if self.cuda_ptr is not None:
            return self.cuda_ptr.read(self.shape, self.dtype)
        raise RuntimeError(f"Cannot read null cuda_ptr of tensor")

    def _write_cuda_memory(self, array: npt.NDArray):
        if self.cuda_ptr is not None:
            self.cuda_ptr.write(array)
        else:
            raise RuntimeError(f"Cannot write data to null cuda_ptr of tensor")

    def __repr__(self):
        if self.device.type == "cpu":
            array = self.cpu_array
        elif self.device.type == "cuda":
            array = self._read_cuda_memory()
        array_str = np.array2string(array, threshold=50, separator=", ")
        return f'tensor({array_str}, dtype={self.dtype}, device="{self.device}")'

    def fill_(self, value):
        from mytorch.ops.elementwise_ops import _fill

        _fill(self, value)

    def copy_(self, tensor):
        from mytorch.ops.broadcast_binary_ops import _copy

        if not isinstance(tensor, Tensor) or self.dtype != tensor.dtype:
            raise RuntimeError(f"Invalid tensor type to copy from: {tensor}")
        _copy(self, tensor.to(self.device))

    def sum(self, dim=None, keepdim=False) -> "Tensor":
        from mytorch.ops.reduce_ops import sum as func

        return func(self, dim, keepdim)

    def mean(self, dim=None, keepdim=False) -> "Tensor":
        from mytorch.ops.reduce_ops import mean as func

        return func(self, dim, keepdim)

    def var(self, dim=None, correction=1, keepdim=False) -> "Tensor":
        from mytorch.ops.reduce_ops import var as func

        return func(self, dim, correction, keepdim)

    def std(self, dim=None, correction=1, keepdim=False) -> "Tensor":
        from mytorch.ops.reduce_ops import std as func

        return func(self, dim, correction, keepdim)

    def reshape(self, shape) -> "Tensor":
        from mytorch.ops.basic_ops import reshape as func

        return func(self, shape)

    def _cast_other_to_tensor(self, other):
        import numbers

        if isinstance(other, numbers.Number):
            other = Tensor(
                cpu_array=np.array(other, dtype=self.dtype.np_dtype), device=self.device
            )
        elif isinstance(other, np.ndarray):
            other = Tensor(
                cpu_array=other.astype(self.dtype.np_dtype), device=self.device
            )
        elif isinstance(other, Tensor):
            pass
        else:
            raise InvalidDataTypeError(type(other))

        return other

    def __add__(self, other) -> "Tensor":
        from mytorch.ops.broadcast_binary_ops import add

        return add(self, self._cast_other_to_tensor(other), 1)

    def __radd__(self, other) -> "Tensor":
        from mytorch.ops.broadcast_binary_ops import add

        return add(self._cast_other_to_tensor(other), self, 1)

    def __sub__(self, other) -> "Tensor":
        from mytorch.ops.broadcast_binary_ops import sub

        return sub(self, self._cast_other_to_tensor(other), 1)

    def __rsub__(self, other) -> "Tensor":
        from mytorch.ops.broadcast_binary_ops import sub

        return sub(self._cast_other_to_tensor(other), self, 1)

    def __mul__(self, other) -> "Tensor":
        from mytorch.ops.broadcast_binary_ops import mul

        return mul(self, self._cast_other_to_tensor(other))

    def __rmul__(self, other) -> "Tensor":
        from mytorch.ops.broadcast_binary_ops import mul

        return mul(self._cast_other_to_tensor(other), self)

    def __truediv__(self, other) -> "Tensor":
        from mytorch.ops.broadcast_binary_ops import div

        return div(self, self._cast_other_to_tensor(other))

    def __rtruediv__(self, other) -> "Tensor":
        from mytorch.ops.broadcast_binary_ops import div

        return div(self._cast_other_to_tensor(other), self)

    def __pow__(self, other) -> "Tensor":
        from mytorch.ops.broadcast_binary_ops import pow

        return pow(self, self._cast_other_to_tensor(other))

    def __rpow__(self, other) -> "Tensor":
        from mytorch.ops.broadcast_binary_ops import pow

        return pow(self._cast_other_to_tensor(other), self)

    def normal_(self, mean, std):
        from mytorch.ops.elementwise_ops import _normal
        from mytorch.rand_generator import RandGenerator

        _normal(self, RandGenerator.instance().generate(), mean, std)

    def uniform_(self, a, b):
        from mytorch.ops.elementwise_ops import _uniform
        from mytorch.rand_generator import RandGenerator

        _uniform(self, RandGenerator.instance().generate(), a, b)

    def item(self):
        return self.to("cpu").cpu_array.item()

    def permute(self, dims) -> "Tensor":
        from mytorch.ops.basic_ops import permute as func

        return func(self, dims)

    def max(self, dim=None, keepdim=False):
        from mytorch.ops.max import max as func

        return func(self, dim, keepdim)

    def eq(self, other) -> "Tensor":
        from mytorch.ops.eq import eq as func

        return func(self, self._cast_other_to_tensor(other))

    def detach(self) -> "Tensor":
        return Tensor(tensor=self, requires_grad=False)

    def numpy(self) -> npt.NDArray:
        if self.requires_grad:
            raise RuntimeError(
                "Tensor that requires gradient cannot be converted to NumPy array"
            )
        return self._numpy()

    def _numpy(self) -> npt.NDArray:
        if self.cpu_array is None:
            raise RuntimeError("Tensor that not on CPU cannot be converted to numpy")
        return self.cpu_array

    def _raw_cuda_ptr(self) -> int:
        if self.cuda_ptr is None:
            raise RuntimeError("Tensor that not on GPU does not have CUDA memory")
        return int(self.cuda_ptr.ptr)

    def backward(self):
        instance = DAGTracker.instance()
        instance.backward(self)
