import numpy as np
import numpy.typing as npt


class DType:
    def __init__(
        self, name: str, np_dtype: npt.DTypeLike, cuda_dtype: str, is_floating: bool
    ):
        self.name = name
        self.np_dtype = np_dtype
        self.cuda_dtype = cuda_dtype
        self.is_floating = is_floating

    def __repr__(self):
        return self.name

    @staticmethod
    def from_np_dtype(np_dtype: npt.DTypeLike) -> "DType":
        for _, v in globals().items():
            if isinstance(v, DType) and v.np_dtype == np_dtype:
                return v
        raise RuntimeError(f"There's no matching mytorch dtype for {np_dtype}")

    @staticmethod
    def from_name(name: str) -> "DType":
        for _, v in globals().items():
            if isinstance(v, DType) and v.name == name:
                return v
        raise RuntimeError(f"There's no matching mytorch dtype for {name}")

    def itemsize(self) -> int:
        return np.dtype(self.np_dtype).itemsize


float16 = DType("float16", np.float16, "half", True)
float32 = DType("float32", np.float32, "float", True)
float64 = DType("float64", np.float64, "double", True)
int8 = DType("int8", np.int8, "int8_t", False)
int16 = DType("int16", np.int16, "int16_t", False)
int32 = DType("int32", np.int32, "int32_t", False)
int64 = DType("int64", np.int64, "int64_t", False)
uint8 = DType("uint8", np.uint8, "uint8_t", False)
uint16 = DType("uint16", np.uint16, "uint16_t", False)
uint32 = DType("uint32", np.uint32, "uint32_t", False)
uint64 = DType("uint64", np.uint64, "uint64_t", False)
