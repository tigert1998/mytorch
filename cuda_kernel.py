import numpy as np
import os.path as osp

from cuda.bindings import driver

from tensor import Tensor
from cuda_utils import check_cuda_errors, CudaStream, CudaCompiler, CudaContextManager


class CudaKernel:
    def __init__(self, kernel, stream: CudaStream):
        self.kernel = kernel
        self.stream = stream

    def _prepare_args(self, args):
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
