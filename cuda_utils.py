import numpy as np

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
        self._cuda_context_manager = CudaContextManager()

    def arch(self, device_id):
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
        major, minor = self.arch(device_id)
        arch_arg = bytes(f"--gpu-architecture=compute_{major}{minor}", "ascii")
        opts = [b"--fmad=false", arch_arg]
        try:
            check_cuda_errors(nvrtc.nvrtcCompileProgram(prog, 2, opts))
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
