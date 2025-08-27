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

    def compile(self, path, device_id, source=None):
        if source is not None:
            content = source
        else:
            with open(path, "r") as f:
                content = f.read()
        prog = check_cuda_errors(
            nvrtc.nvrtcCreateProgram(content.encode(), path.encode(), 0, [], [])
        )
        major, minor = self._arch(device_id)
        cuda_path = os.environ["CUDA_PATH"]
        cuda_include_paths = [
            osp.join(cuda_path, "include"),
            osp.join(cuda_path, "include/cccl"),
            osp.join(osp.dirname(__file__), "../cuda_kernels/include"),
        ]
        opts = [
            b"--fmad=false",
            f"--gpu-architecture=compute_{major}{minor}".encode(),
            *[f"-I{i}".encode() for i in cuda_include_paths],
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

    def run(self, grid_dim, block_dim, args, num_shared_bytes=0):
        self.stream.set_device()
        args = self._prepare_args(args)
        ptr_array = np.array([i.ctypes.data for i in args], dtype=np.uint64)
        check_cuda_errors(
            driver.cuLaunchKernel(
                self.kernel,
                *grid_dim,
                *block_dim,
                num_shared_bytes,
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

    def get_kernel(self, cu_file_path, func_name, device_id, source=None):
        stream = self.get_stream(device_id)
        stream.set_device()
        if self._modules.get(device_id, {}).get(cu_file_path) is None:
            ptx = self._cuda_compiler.compile(
                osp.join(osp.dirname(__file__), "../cuda_kernels", cu_file_path),
                device_id,
                source=source,
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
        from mytorch.cuda.memory_allocator import SimpleCudaMemoryAllocator

        self.context_manager = CudaContextManager()
        self.compiler = CudaCompiler(self.context_manager)
        self.kernel_and_stream_manager = CudaKernelAndStreamManager(
            self.compiler, self.context_manager
        )
        self.allocator = SimpleCudaMemoryAllocator()

    def destroy(self):
        self.allocator.destroy()
        self.kernel_and_stream_manager.destroy()
        self.context_manager.destroy()

    def __del__(self):
        self.destroy()
