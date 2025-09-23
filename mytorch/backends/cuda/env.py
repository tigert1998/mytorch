import os
import numpy as np
import ctypes
import os.path as osp
from glob import glob
import sys
import regex as re
import json

import numpy.typing as npt
from cuda.bindings import driver, nvrtc, runtime

from mytorch.backends.backend_dispatcher import BackendDispatcher


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


class CudaTimer:
    def __init__(self, stream: CudaStream):
        self.stream = stream
        self._start = check_cuda_errors(runtime.cudaEventCreate())
        self._end = check_cuda_errors(runtime.cudaEventCreate())

    def start(self):
        check_cuda_errors(runtime.cudaEventRecord(self._start, self.stream.stream))

    def end(self) -> float:
        check_cuda_errors(runtime.cudaEventRecord(self._end, self.stream.stream))
        check_cuda_errors(runtime.cudaEventSynchronize(self._end))
        return check_cuda_errors(runtime.cudaEventElapsedTime(self._start, self._end))

    def __del__(self):
        check_cuda_errors(runtime.cudaEventDestroy(self._start))
        check_cuda_errors(runtime.cudaEventDestroy(self._end))


class CudaSourceGenerator:
    def __init__(self):
        self.kernel_src_path = osp.join(osp.dirname(__file__), "../../native/cuda")
        self.generated_src_path = osp.join(
            osp.dirname(__file__), "../../../build/generated"
        )

    def _get_templated_source(self, path: str) -> tuple[str, str]:
        from mytorch.dtype import DType

        path = osp.join(self.kernel_src_path, path)
        vars_path = osp.join(path, "vars.json")
        if osp.isfile(vars_path):
            # concate into a source from *.template
            content = ""
            with open(vars_path) as f:
                template_vars = json.loads(f.read())
            for template_path in sorted(glob(osp.join(path, "*.template"))):
                with open(template_path, "r") as f:
                    template_content = f.read()
                    for dic in template_vars:
                        try:
                            content += template_content.format(**dic)
                        except:
                            pass
            hint_path = path
        else:
            hint_path = osp.join(path, "source.cu")
            # read source from source.cu
            with open(hint_path) as f:
                content = f.read()

        with open(osp.join(path, "instantiation.json")) as f:
            dic = json.load(f)
            instantiation = {
                key: [
                    list(map(DType.from_name, types.split(" "))) for types in dic[key]
                ]
                for key in dic.keys()
            }

        pattern = r"""
            template\s*<[^>]+>\s* # template <typename T>
            (?:\w+__\s+)*       # __global__
            \w+\s+              # void
            \w+\s*              # function name
            \(.*?\)             # argument types
        """
        template_funcs = re.findall(pattern, content, re.DOTALL | re.VERBOSE)
        pattern = r"""
            template\s*<([^>]+)>\s*    # capture template T
            ((?:__\w+__\s*)*)          # capture __global__
            (\w+)\s+                   # capture return type
            (\w+)\s*                   # capture funtion name
            \(\s*([^)]*)\s*\)          # capture arguments
        """

        def replace_arg_types_with_template(arg_types, dtypes):
            new_arg_types = []
            for arg_type in arg_types:
                is_match = False
                for i, template_type in enumerate(template_types):
                    pattern = f"(?:\\w|^){template_type}(?:\\W|$)"
                    if re.match(pattern, arg_type):
                        new_arg_types.append(
                            arg_type.replace(
                                template_type,
                                dtypes[i].cuda_dtype,
                            )
                        )
                        is_match = True
                        break
                if not is_match:
                    new_arg_types.append(arg_type)
            return new_arg_types

        content += """
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif
"""

        for template_func in template_funcs:
            match = re.match(pattern, template_func, re.VERBOSE | re.DOTALL)
            template_part = match.group(1)
            template_types = re.findall(r"typename\s+(\w+)", template_part)
            has_global = match.group(2)
            return_type = match.group(3)
            function_name = match.group(4)
            params_str = match.group(5)

            param_pattern = r"(?:\s*,|^)([^,]+[&|*|\s+])(\w+)"
            params = re.findall(param_pattern, params_str)

            arg_types = [p[0].strip() for p in params]
            arg_names = [p[1].strip() for p in params]

            if function_name not in instantiation:
                continue
            if len(instantiation[function_name][0]) != len(template_types):
                raise RuntimeError("Template variables number mismatch")

            for dtypes in instantiation[function_name]:
                new_arg_types = replace_arg_types_with_template(arg_types, dtypes)
                desc = (
                    has_global
                    if has_global is not None and has_global.strip() != ""
                    else "API"
                )
                func_decl = (
                    f'extern "C" {desc} {return_type} {function_name}'
                    + "".join([f"_{dtype.name}" for dtype in dtypes])
                    + "("
                    + ", ".join(
                        [
                            f"{arg_type} {arg_name}"
                            for arg_type, arg_name in zip(new_arg_types, arg_names)
                        ]
                    )
                    + ")"
                )
                func_body = f"{function_name}(" + ", ".join(arg_names) + ");"
                content += func_decl + "{\n  " + func_body + "\n}\n"

        return content, hint_path

    def save_templated_source(self, path):
        generated_path = osp.join(self.generated_src_path, path)
        os.makedirs(osp.dirname(generated_path), exist_ok=True)
        with open(generated_path, "w") as f:
            f.write(self._get_templated_source(path)[0])


class CudaCompiler:
    _cuda_context_manager: CudaContextManager
    cudadevrt_path: str

    def __init__(self, cuda_context_manager):
        self._cuda_source_generator = CudaSourceGenerator()
        self._cuda_context_manager = cuda_context_manager
        cuda_path = os.environ["CUDA_PATH"]
        cudadevrt_paths = glob(f"{cuda_path}/lib*/**/*cudadevrt.*", recursive=True)
        if len(cudadevrt_paths) != 1:
            raise RuntimeError(f"cudadevrt path is vague: {cudadevrt_paths}")
        self.cudadevrt_path = cudadevrt_paths[0]

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

    def compile(self, path, device_id) -> driver.CUmodule:
        if osp.isdir(path):
            content, path = self._cuda_source_generator._get_templated_source(path)
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
            self._cuda_source_generator.kernel_src_path,
        ]
        opts = [
            b"--fmad=false",
            f"--gpu-architecture=compute_{major}{minor}".encode(),
            b"-dc",
            *[f"-I{i}".encode() for i in cuda_include_paths],
        ]
        try:
            check_cuda_errors(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
        except Exception:
            log_size = check_cuda_errors(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * log_size
            check_cuda_errors(nvrtc.nvrtcGetProgramLog(prog, log))
            raise RuntimeError(f"Cuda compile error: {log.decode()}")
        ptx_size = check_cuda_errors(nvrtc.nvrtcGetPTXSize(prog))
        ptx = b" " * ptx_size
        check_cuda_errors(nvrtc.nvrtcGetPTX(prog, ptx))
        error_log_size = 8192
        error_log = b" " * error_log_size
        link_options = {
            driver.CUjit_option.CU_JIT_OPTIMIZATION_LEVEL: 3,
            driver.CUjit_option.CU_JIT_TARGET: driver.CUjit_target(major * 10 + minor),
            driver.CUjit_option.CU_JIT_ERROR_LOG_BUFFER: error_log,
            driver.CUjit_option.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: error_log_size,
        }
        link_state = check_cuda_errors(
            driver.cuLinkCreate(
                len(link_options),
                list(link_options.keys()),
                list(link_options.values()),
            )
        )
        check_cuda_errors(
            driver.cuLinkAddData(
                link_state,
                driver.CUjitInputType.CU_JIT_INPUT_PTX,
                ptx,
                ptx_size,
                f"{path}".encode(),
                0,
                [],
                [],
            )
        )
        for lib in [self.cudadevrt_path]:
            check_cuda_errors(
                driver.cuLinkAddFile(
                    link_state,
                    driver.CUjitInputType.CU_JIT_INPUT_LIBRARY,
                    lib.encode(),
                    0,
                    [],
                    [],
                )
            )
        try:
            cubin_ptr, cubin_size = check_cuda_errors(driver.cuLinkComplete(link_state))
        except:
            raise RuntimeError(f"Cuda link error: {error_log.decode()}")
        self._cuda_context_manager.set_device(device_id)
        module = check_cuda_errors(driver.cuModuleLoadData(cubin_ptr))
        check_cuda_errors(driver.cuLinkDestroy(link_state))
        return module


class CudaLibrary:
    def __init__(self):
        if sys.platform == "win32":
            path = "build/Release/mytorch-cuda-backend.dll"
        else:
            path = "build/Release/libmytorch-cuda-backend.so"
        self._lib = ctypes.cdll.LoadLibrary(path)

    def _prepare_args(self, args):
        from mytorch.tensor import Tensor

        new_args = []
        argtypes = []

        for arg in args:
            if isinstance(arg, Tensor):
                new_args.append(int(arg._native().ptr))
                argtypes.append(ctypes.c_void_p)
            elif isinstance(arg, np.ndarray):
                new_args.append(arg.item())
                if arg.dtype == np.float32:
                    argtypes.append(ctypes.c_float)
                elif arg.dtype == np.int32:
                    argtypes.append(ctypes.c_int)
                elif arg.dtype == np.int8:
                    argtypes.append(ctypes.c_int8)
                else:
                    raise RuntimeError(
                        f"Invalid argument type {arg.dtype} for CUDA library"
                    )
            elif arg is None:
                new_args.append(0)
                argtypes.append(ctypes.c_void_p)
            elif isinstance(arg, CudaStream):
                new_args.append(int(arg.stream))
                argtypes.append(ctypes.c_void_p)
            else:
                raise RuntimeError(
                    f"Invalid data type for invoking kernel: {arg} ({type(arg)})"
                )
        return new_args, argtypes

    def run(self, function_name, args, stream: CudaStream, timer: bool = False):
        args, argtypes = self._prepare_args(args + [stream])
        self._lib.batch_norm2d_reference_float32.argtypes = argtypes
        func = self._lib[function_name]
        func.argtypes = argtypes
        if timer:
            cuda_timer = CudaTimer(stream)
            cuda_timer.start()
        func(*args)
        if timer:
            return cuda_timer.end()


class CudaKernel:
    def __init__(self, kernel, stream: CudaStream):
        self.kernel = kernel
        self.stream = stream

    def _prepare_args(self, args):
        from mytorch.tensor import Tensor

        np_args = []
        for arg in args:
            if isinstance(arg, Tensor):
                if (
                    arg.device.type != "cuda"
                    or arg.device.index != self.stream.device_id
                ):
                    raise RuntimeError(
                        f"Invalid device for invoking CUDA kernel: {arg.device}"
                    )
                np_args.append(np.array([int(arg._native().ptr)], dtype=np.uint64))
            elif isinstance(arg, np.ndarray):
                np_args.append(arg)
            elif arg is None:
                np_args.append(np.array(0, dtype=np.uint64))
            else:
                raise RuntimeError(
                    f"Invalid data type for invoking kernel: {arg} ({type(arg)})"
                )
        return np_args

    def run(self, grid_dim, block_dim, args, num_shared_bytes=0, timer: bool = False):
        self.stream.set_device()
        args = self._prepare_args(args)
        ptr_array = np.array([i.ctypes.data for i in args], dtype=np.uint64)
        if timer:
            cuda_timer = CudaTimer(self.stream)
            cuda_timer.start()
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
        if timer:
            return cuda_timer.end()


class CudaKernelAndStreamManager:
    _cuda_compiler: CudaCompiler
    _cuda_context_manager: CudaContextManager
    _streams: dict[int, CudaStream]
    _modules: dict[int, dict[str, driver.CUmodule]]

    def __init__(
        self, cuda_compiler: CudaCompiler, cuda_context_manager: CudaContextManager
    ):
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
            if self._modules.get(device_id) is None:
                self._modules[device_id] = {}
            self._modules[device_id][cu_file_path] = self._cuda_compiler.compile(
                osp.join(
                    self._cuda_compiler._cuda_source_generator.kernel_src_path,
                    cu_file_path,
                ),
                device_id,
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
        from mytorch.backends.cuda.memory_allocator import SimpleCudaMemoryAllocator

        self.context_manager = CudaContextManager()
        self.compiler = CudaCompiler(self.context_manager)
        self.kernel_and_stream_manager = CudaKernelAndStreamManager(
            self.compiler, self.context_manager
        )
        self.allocator = SimpleCudaMemoryAllocator()
        self.library = CudaLibrary()

    def destroy(self):
        self.allocator.destroy()
        self.kernel_and_stream_manager.destroy()
        self.context_manager.destroy()

    def __del__(self):
        self.destroy()


class CudaMemory:
    def __init__(self, device_id, size):
        self.device_id = device_id
        self.size = size
        CudaEnv.instance().context_manager.set_device(self.device_id)
        self.ptr = CudaEnv.instance().allocator.allocate(size)

    def destroy(self):
        # you can call destory in advance
        if self.ptr is not None:
            CudaEnv.instance().context_manager.set_device(self.device_id)
            CudaEnv.instance().allocator.deallocate(self.ptr)
        self.ptr = None

    def __del__(self):
        self.destroy()

    def read(self, shape, dtype: npt.DTypeLike) -> npt.NDArray:
        CudaEnv.instance().context_manager.set_device(self.device_id)
        array = np.zeros(shape=shape, dtype=dtype)
        check_cuda_errors(
            driver.cuMemcpyDtoH(
                array.ctypes.data,
                self.ptr,
                array.itemsize * array.size,
            )
        )
        return array

    def write(self, array: npt.NDArray):
        CudaEnv.instance().context_manager.set_device(self.device_id)
        check_cuda_errors(
            driver.cuMemcpyHtoD(
                self.ptr,
                array.ctypes.data,
                array.itemsize * array.size,
            )
        )


@BackendDispatcher.instance().register_backend_function("cuda", "allocate_memory")
def cuda_allocate_memory(device_id: int, size: int) -> CudaMemory:
    return CudaMemory(device_id, size)


@BackendDispatcher.instance().register_backend_function("cuda", "transfer_memory")
def transfer_memory(mem0: CudaMemory, mem1: CudaMemory):
    # mem0 = mem1
    check_cuda_errors(
        driver.cuMemcpyPeer(
            mem0.ptr,
            check_cuda_errors(driver.cuDeviceGet(mem0.device_id)),
            mem1.ptr,
            check_cuda_errors(driver.cuDeviceGet(mem1.device_id)),
            mem0.size,
        )
    )
