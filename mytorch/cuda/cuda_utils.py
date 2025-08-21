import os
import numpy as np
import os.path as osp
import ctypes
from typing import Optional

from cuda.bindings import driver, nvrtc
import cuda.pathfinder


def _cuda_get_error_enum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name.decode() if err == driver.CUresult.CUDA_SUCCESS else "Unknown"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1].decode()
    else:
        raise RuntimeError("Unknown error type: {}".format(error))


def check_cuda_errors(result):
    if result[0].value:
        raise RuntimeError(
            "CUDA error code={}({})".format(
                result[0].value, _cuda_get_error_enum(result[0])
            )
        )
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


class CudaContextManager:
    def __init__(self):
        check_cuda_errors(driver.cuInit(0))
        self._device_id_to_context = {}

    def set_device(self, device_id):
        if self._device_id_to_context.get(device_id) is None:
            cu_device = check_cuda_errors(driver.cuDeviceGet(device_id))
            context = check_cuda_errors(driver.cuCtxCreate(None, 0, cu_device))
            self._device_id_to_context[device_id] = context

        return check_cuda_errors(
            driver.cuCtxSetCurrent(self._device_id_to_context[device_id])
        )

    def destroy(self):
        for key in self._device_id_to_context.keys():
            check_cuda_errors(driver.cuCtxDestroy(self._device_id_to_context[key]))


class CudaStream:
    def __init__(self, device_id, cuda_context_manager):
        self.cuda_context_manager = cuda_context_manager
        self.device_id = device_id
        self.set_device()
        self.stream = check_cuda_errors(driver.cuStreamCreate(0))

    def set_device(self):
        self.cuda_context_manager.set_device(self.device_id)

    def sync(self):
        check_cuda_errors(driver.cuStreamSynchronize(self.stream))

    def destroy(self):
        check_cuda_errors(driver.cuStreamDestroy(self.stream))


class CudaCompiler:
    def __init__(self, cuda_context_manager):
        self._cuda_context_manager = cuda_context_manager

    def _arch(self, device_id):
        cu_device = check_cuda_errors(driver.cuDeviceGet(device_id))
        major = check_cuda_errors(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                cu_device,
            )
        )
        minor = check_cuda_errors(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                cu_device,
            )
        )
        return major, minor

    def compile(self, path, device_id):
        with open(path, "r") as f:
            content = f.read()
        prog = check_cuda_errors(
            nvrtc.nvrtcCreateProgram(content.encode(), path.encode(), 0, [], [])
        )
        major, minor = self._arch(device_id)
        cuda_include_path = os.path.join(os.environ["CUDA_PATH"], "include")
        opts = [
            b"--fmad=false",
            f"--gpu-architecture=compute_{major}{minor}".encode(),
            f"-I{cuda_include_path}".encode(),
        ]
        try:
            check_cuda_errors(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
        except Exception:
            log_size = check_cuda_errors(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * log_size
            check_cuda_errors(nvrtc.nvrtcGetProgramLog(prog, log))
            raise RuntimeError(f"Cuda compile error: {log.decode()}")
        ptr_size = check_cuda_errors(nvrtc.nvrtcGetPTXSize(prog))
        ptx = b" " * ptr_size
        check_cuda_errors(nvrtc.nvrtcGetPTX(prog, ptx))
        ptx = np.char.array(ptx)
        return ptx


class CudaKernel:
    def __init__(self, kernel, stream: CudaStream):
        self.kernel = kernel
        self.stream = stream

    def _prepare_args(self, args):
        from mytorch.tensor import Tensor

        np_args = []
        for arg in args:
            if isinstance(arg, Tensor):
                assert (
                    arg.device.type == "cuda"
                    and arg.device.index == self.stream.device_id
                )
                np_args.append(np.array([int(arg.cuda_ptr.ptr)], dtype=np.uint64))
            elif isinstance(arg, np.ndarray):
                np_args.append(arg)
            elif arg is None:
                np_args.append(np.array(0, dtype=np.uint64))
            else:
                error_msg = (
                    f"Invalid data type for invoking kernel: {arg} ({type(arg)})"
                )
                assert False, error_msg
        return np_args

    def run(self, grid_dim, block_dim, args):
        self.stream.set_device()
        args = self._prepare_args(args)
        ptr_array = np.array([i.ctypes.data for i in args], dtype=np.uint64)
        check_cuda_errors(
            driver.cuLaunchKernel(
                self.kernel,
                *grid_dim,
                *block_dim,
                0,
                self.stream.stream,
                ptr_array.ctypes.data,
                0,
            )
        )


class CudaKernelAndStreamManager:
    def __init__(self, cuda_compiler, cuda_context_manager):
        self._cuda_compiler = cuda_compiler
        self._cuda_context_manager = cuda_context_manager
        self._streams = {}
        self._modules = {}

    def destroy(self):
        for device_id, dic in self._modules.items():
            self._cuda_context_manager.set_device(device_id)
            for module in dic.values():
                check_cuda_errors(driver.cuModuleUnload(module))

        for device_id in self._streams.keys():
            self._streams[device_id].destroy()

    def get_stream(self, device_id) -> CudaStream:
        if self._streams.get(device_id) is None:
            self._streams[device_id] = CudaStream(device_id, self._cuda_context_manager)
        return self._streams[device_id]

    def get_kernel(self, cu_file_path, func_name, device_id):
        stream = self.get_stream(device_id)
        stream.set_device()
        if self._modules.get(device_id, {}).get(cu_file_path) is None:
            ptx = self._cuda_compiler.compile(
                osp.join(osp.dirname(__file__), "../cuda_kernels", cu_file_path),
                device_id,
            )
            if self._modules.get(device_id) is None:
                self._modules[device_id] = {}
            self._modules[device_id][cu_file_path] = check_cuda_errors(
                driver.cuModuleLoadData(ptx.ctypes.data)
            )
        module = self._modules[device_id][cu_file_path]
        kernel = check_cuda_errors(
            driver.cuModuleGetFunction(module, func_name.encode())
        )
        return CudaKernel(kernel, stream)


class CudaEnv:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = CudaEnv()
        return cls._instance

    def __init__(self):
        self.context_manager = CudaContextManager()
        self.compiler = CudaCompiler(self.context_manager)
        self.kernel_and_stream_manager = CudaKernelAndStreamManager(
            self.compiler, self.context_manager
        )

    def destroy(self):
        self.kernel_and_stream_manager.destroy()
        self.context_manager.destroy()

    def __del__(self):
        self.destroy()


class CublasLt:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = CublasLt()
        return cls._instance

    def __init__(self):
        cublas_lt = cuda.pathfinder.load_nvidia_dynamic_lib("cublasLt")
        lib = ctypes.CDLL(name="cublasLt", handle=cublas_lt._handle_uint)
        lib.cublasLtMatmul.argtypes = [
            ctypes.c_void_p,  # light_handle
            ctypes.c_void_p,  # compute_desc
            ctypes.c_void_p,  # alpha
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # a_desc
            ctypes.c_void_p,  # B
            ctypes.c_void_p,
            ctypes.c_void_p,  # beta
            ctypes.c_void_p,  # C
            ctypes.c_void_p,
            ctypes.c_void_p,  # D
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,  # stream
        ]
        lib.cublasLtMatmul.restype = ctypes.c_int

        lib.cublasLtCreate.argtypes = [ctypes.c_void_p]
        lib.cublasLtCreate.restype = ctypes.c_int

        lib.cublasLtMatmulDescCreate.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.cublasLtMatmulDescCreate.restype = ctypes.c_int

        lib.cublasLtMatmulDescSetAttribute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.cublasLtMatmulDescSetAttribute.restype = ctypes.c_int

        lib.cublasLtMatrixLayoutCreate.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int64,
        ]
        lib.cublasLtMatrixLayoutCreate.restype = ctypes.c_int

        lib.cublasLtMatrixLayoutSetAttribute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.cublasLtMatrixLayoutSetAttribute.restype = ctypes.c_int

        lib.cublasLtMatrixTransformDescCreate.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.cublasLtMatrixTransformDescCreate.restype = ctypes.c_int

        lib.cublasLtMatrixTransform.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_void_p,  # transform_desc
            ctypes.c_void_p,  # alpha
            ctypes.c_void_p,  # a
            ctypes.c_void_p,  # a_desc
            ctypes.c_void_p,  # beta
            ctypes.c_void_p,  # b
            ctypes.c_void_p,  # b_desc
            ctypes.c_void_p,  # c
            ctypes.c_void_p,  # c_desc
            ctypes.c_void_p,  # stream
        ]
        lib.cublasLtMatrixTransform.restype = ctypes.c_int

        lib.cublasLtDestroy.argtypes = [ctypes.c_void_p]
        lib.cublasLtDestroy.restype = ctypes.c_int

        lib.cublasLtMatrixLayoutDestroy.argtypes = [ctypes.c_void_p]
        lib.cublasLtMatrixLayoutDestroy.restype = ctypes.c_int

        lib.cublasLtMatmulDescDestroy.argtypes = [ctypes.c_void_p]
        lib.cublasLtMatmulDescDestroy.restype = ctypes.c_int

        self._lib = lib

        # cublasLtMatmulDescAttributes_t
        self.CUBLASLT_MATMUL_DESC_COMPUTE_TYPE = 0
        self.CUBLASLT_MATMUL_DESC_SCALE_TYPE = 1
        self.CUBLASLT_MATMUL_DESC_POINTER_MODE = 2
        self.CUBLASLT_MATMUL_DESC_TRANSA = 3
        self.CUBLASLT_MATMUL_DESC_TRANSB = 4
        self.CUBLASLT_MATMUL_DESC_TRANSC = 5
        self.CUBLASLT_MATMUL_DESC_FILL_MODE = 6
        self.CUBLASLT_MATMUL_DESC_EPILOGUE = 7
        self.CUBLASLT_MATMUL_DESC_BIAS_POINTER = 8
        self.CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE = 10
        self.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER = 11
        self.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD = 12
        self.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE = 13
        self.CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE = 14
        self.CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET = 15
        self.CUBLASLT_MATMUL_DESC_A_SCALE_POINTER = 17
        self.CUBLASLT_MATMUL_DESC_B_SCALE_POINTER = 18
        self.CUBLASLT_MATMUL_DESC_C_SCALE_POINTER = 19
        self.CUBLASLT_MATMUL_DESC_D_SCALE_POINTER = 20
        self.CUBLASLT_MATMUL_DESC_AMAX_D_POINTER = 21
        self.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE = 22
        self.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER = 23
        self.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER = 24
        self.CUBLASLT_MATMUL_DESC_FAST_ACCUM = 25
        self.CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = 26
        self.CUBLASLT_MATMUL_DESC_A_SCALE_MODE = 31
        self.CUBLASLT_MATMUL_DESC_B_SCALE_MODE = 32
        self.CUBLASLT_MATMUL_DESC_C_SCALE_MODE = 33
        self.CUBLASLT_MATMUL_DESC_D_SCALE_MODE = 34
        self.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_MODE = 35
        self.CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER = 36
        self.CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE = 37

        # cublasOperation_t
        self.CUBLAS_OP_N = 0
        self.CUBLAS_OP_T = 1
        self.CUBLAS_OP_C = 2
        self.CUBLAS_OP_HERMITAN = 2
        self.CUBLAS_OP_CONJG = 3

        # cublasLtOrder_t
        self.CUBLASLT_ORDER_COL = 0
        self.CUBLASLT_ORDER_ROW = 1
        self.CUBLASLT_ORDER_COL32 = 2
        self.CUBLASLT_ORDER_COL4_4R2_8C = 3
        self.CUBLASLT_ORDER_COL32_2R_4R4 = 4

        # cublasLtMatrixLayoutAttribute_t
        self.CUBLASLT_MATRIX_LAYOUT_TYPE = 0
        self.CUBLASLT_MATRIX_LAYOUT_ORDER = 1
        self.CUBLASLT_MATRIX_LAYOUT_ROWS = 2
        self.CUBLASLT_MATRIX_LAYOUT_COLS = 3
        self.CUBLASLT_MATRIX_LAYOUT_LD = 4
        self.CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 5
        self.CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 6
        self.CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET = 7
        self.CUBLASLT_MATRIX_LAYOUT_BATCH_MODE = 8

        # cublasComputeType_t
        self.CUBLAS_COMPUTE_16F = 64
        self.CUBLAS_COMPUTE_16F_PEDANTIC = 65
        self.CUBLAS_COMPUTE_32F = 68
        self.CUBLAS_COMPUTE_32F_PEDANTIC = 69
        self.CUBLAS_COMPUTE_32F_FAST_16F = 74
        self.CUBLAS_COMPUTE_32F_FAST_16BF = 75
        self.CUBLAS_COMPUTE_32F_FAST_TF32 = 77
        self.CUBLAS_COMPUTE_32F_EMULATED_16BFX9 = 78
        self.CUBLAS_COMPUTE_64F = 70
        self.CUBLAS_COMPUTE_64F_PEDANTIC = 71
        self.CUBLAS_COMPUTE_32I = 72
        self.CUBLAS_COMPUTE_32I_PEDANTIC = 73

        # cudaDataType
        self.CUDA_R_16F = 2
        self.CUDA_C_16F = 6
        self.CUDA_R_16BF = 14
        self.CUDA_C_16BF = 15
        self.CUDA_R_32F = 0
        self.CUDA_C_32F = 4
        self.CUDA_R_64F = 1
        self.CUDA_C_64F = 5
        self.CUDA_R_4I = 16
        self.CUDA_C_4I = 17
        self.CUDA_R_4U = 18
        self.CUDA_C_4U = 19
        self.CUDA_R_8I = 3
        self.CUDA_C_8I = 7
        self.CUDA_R_8U = 8
        self.CUDA_C_8U = 9
        self.CUDA_R_16I = 20
        self.CUDA_C_16I = 21
        self.CUDA_R_16U = 22
        self.CUDA_C_16U = 23
        self.CUDA_R_32I = 10
        self.CUDA_C_32I = 11
        self.CUDA_R_32U = 12
        self.CUDA_C_32U = 13
        self.CUDA_R_64I = 24
        self.CUDA_C_64I = 25
        self.CUDA_R_64U = 26
        self.CUDA_C_64U = 27
        self.CUDA_R_8F_E4M3 = 28
        self.CUDA_R_8F_UE4M3 = self.CUDA_R_8F_E4M3
        self.CUDA_R_8F_E5M2 = 29
        self.CUDA_R_8F_UE8M0 = 30
        self.CUDA_R_6F_E2M3 = 31
        self.CUDA_R_6F_E3M2 = 32
        self.CUDA_R_4F_E2M1 = 33

        self.cublas_status = {}
        self.cublas_status[0] = "CUBLAS_STATUS_SUCCESS"
        self.cublas_status[1] = "CUBLAS_STATUS_NOT_INITIALIZED"
        self.cublas_status[3] = "CUBLAS_STATUS_ALLOC_FAILED"
        self.cublas_status[7] = "CUBLAS_STATUS_INVALID_VALUE"
        self.cublas_status[8] = "CUBLAS_STATUS_ARCH_MISMATCH"
        self.cublas_status[11] = "CUBLAS_STATUS_MAPPING_ERROR"
        self.cublas_status[13] = "CUBLAS_STATUS_EXECUTION_FAILED"
        self.cublas_status[14] = "CUBLAS_STATUS_INTERNAL_ERROR"
        self.cublas_status[15] = "CUBLAS_STATUS_NOT_SUPPORTED"
        self.cublas_status[16] = "CUBLAS_STATUS_LICENSE_ERROR"

    def _check_cublas_errors(self, ret):
        assert ret == 0, self.cublas_status[ret]

    def create(self) -> np.ndarray:
        light_handle = np.array(0, dtype=np.uint64)
        self._check_cublas_errors(self._lib.cublasLtCreate(light_handle.ctypes.data))
        return light_handle

    def destroy(self, light_handle: np.ndarray):
        self._check_cublas_errors(self._lib.cublasLtDestroy(light_handle.item()))

    def matrix_layout_destroy(self, mat_layout: np.ndarray):
        self._check_cublas_errors(
            self._lib.cublasLtMatrixLayoutDestroy(mat_layout.item())
        )

    def matmul_desc_destroy(self, matmul_desc: np.ndarray):
        self._check_cublas_errors(
            self._lib.cublasLtMatmulDescDestroy(matmul_desc.item())
        )

    def matmul_desc_create(self, compute_type: int, scale_type: int) -> np.ndarray:
        matmul_desc = np.array(0, dtype=np.uint64)
        self._check_cublas_errors(
            self._lib.cublasLtMatmulDescCreate(
                matmul_desc.ctypes.data, compute_type, scale_type
            )
        )
        return matmul_desc

    def matmul_desc_set_attribute(
        self, matmul_desc: np.ndarray, attr: int, data: np.ndarray
    ):
        self._check_cublas_errors(
            self._lib.cublasLtMatmulDescSetAttribute(
                matmul_desc.item(), attr, data.ctypes.data, data.itemsize * data.size
            )
        )

    def matrix_layout_create(
        self, cuda_data_type: int, rows: int, cols: int, ld: int
    ) -> np.ndarray:
        mat_layout = np.array(0, dtype=np.uint64)
        self._check_cublas_errors(
            self._lib.cublasLtMatrixLayoutCreate(
                mat_layout.ctypes.data,
                cuda_data_type,
                rows,
                cols,
                ld,
            ),
        )
        return mat_layout

    def matrix_layout_set_attribute(
        self, mat_layout: np.ndarray, attr: int, data: np.ndarray
    ):
        self._check_cublas_errors(
            self._lib.cublasLtMatrixLayoutSetAttribute(
                mat_layout.item(), attr, data.ctypes.data, data.itemsize * data.size
            )
        )

    def matrix_transform_desc_create(self, scale_type: int) -> np.ndarray:
        desc = np.array(0, dtype=np.uint64)
        self._check_cublas_errors(
            self._lib.cublasLtMatrixTransformDescCreate(desc.ctypes.data, scale_type)
        )
        return desc

    def matrix_transform(
        self,
        handle: np.ndarray,
        transform_desc: np.ndarray,
        alpha: np.ndarray,
        a: int,
        a_desc: np.ndarray,
        beta: np.ndarray,
        b: int,
        b_desc: np.ndarray,
        c: int,
        c_desc: np.ndarray,
        stream: CudaStream,
    ):
        self._check_cublas_errors(
            self._lib.cublasLtMatrixTransform(
                handle.item(),
                transform_desc.item(),
                alpha.ctypes.data,
                a,
                a_desc.item(),
                beta.ctypes.data,
                b,
                b_desc.item(),
                c,
                c_desc.item(),
                stream.stream.getPtr(),
            )
        )

    def matmul(
        self,
        handle: np.ndarray,
        compute_desc: np.ndarray,
        alpha: np.ndarray,
        a: int,
        a_desc: np.ndarray,
        b: int,
        b_desc: np.ndarray,
        beta: np.ndarray,
        c: int,
        c_desc: np.ndarray,
        d: int,
        d_desc: np.ndarray,
        algo: Optional[np.ndarray],
        workspace: int,
        workspace_size_in_bytes: int,
        stream: CudaStream,
    ):
        self._check_cublas_errors(
            self._lib.cublasLtMatmul(
                handle.item(),
                compute_desc.item(),
                alpha.ctypes.data,
                a,
                a_desc.item(),
                b,
                b_desc.item(),
                beta.ctypes.data,
                c,
                c_desc.item(),
                d,
                d_desc.item(),
                0 if algo is None else algo.ctypes.data,
                workspace,
                workspace_size_in_bytes,
                int(stream.stream),
            )
        )
