import os
import numpy as np
import os.path as osp

from cuda.bindings import driver, nvrtc


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
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = CudaContextManager()
        return cls._instance

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

    def __del__(self):
        for key in self._device_id_to_context.keys():
            check_cuda_errors(driver.cuCtxDestroy(self._device_id_to_context[key]))


class CudaStream:
    def __init__(self, device_id):
        self.cuda_context_manager = CudaContextManager.instance()
        self.device_id = device_id
        self.set_device()
        self.stream = check_cuda_errors(driver.cuStreamCreate(0))

    def set_device(self):
        self.cuda_context_manager.set_device(self.device_id)

    def sync(self):
        check_cuda_errors(driver.cuStreamSynchronize(self.stream))

    def __del__(self):
        check_cuda_errors(driver.cuStreamDestroy(self.stream))


class CudaCompiler:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = CudaCompiler()
        return cls._instance

    def __init__(self):
        self._cuda_context_manager = CudaContextManager.instance()

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
        from tensor import Tensor

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
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = CudaKernelAndStreamManager()
        return cls._instance

    def __init__(self):
        self._cuda_compiler = CudaCompiler.instance()
        self._cuda_context_manager = CudaContextManager.instance()
        self._streams = {}
        self._modules = {}

    def __del__(self):
        for device_id, dic in self._modules.items():
            self._cuda_context_manager.set_device(device_id)
            for module in dic.values():
                check_cuda_errors(driver.cuModuleUnload(module))

    def get_stream(self, device_id) -> CudaStream:
        if self._streams.get(device_id) is None:
            self._streams[device_id] = CudaStream(device_id)
        return self._streams[device_id]

    def get_kernel(self, cu_file_path, func_name, device_id):
        stream = self.get_stream(device_id)
        stream.set_device()
        if self._modules.get(device_id, {}).get(cu_file_path) is None:
            ptx = self._cuda_compiler.compile(
                osp.join("cuda_kernels", cu_file_path), device_id
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
