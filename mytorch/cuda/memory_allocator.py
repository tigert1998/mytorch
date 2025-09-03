import numpy as np
from typing import Dict, List

from cuda.bindings import driver
from mytorch.cuda.env import check_cuda_errors


class SimpleCudaMemoryAllocator:
    _pool: Dict[int, List[driver.CUdeviceptr]]
    _ptr_mem_size: Dict[driver.CUdeviceptr, int]

    def __init__(self):
        self._pool = {}
        self._ptr_mem_size = {}

    def allocate(self, size: int) -> driver.CUdeviceptr:
        if size <= 0:
            raise RuntimeError(
                f"SimpleCudaMemoryAllocator cannot allocate {size} bytes"
            )
        new_size = 1 << int(np.ceil(np.log2(size)))
        new_size = max(new_size, 8)  # at least 8 bytes
        if self._pool.get(new_size) is None:
            self._pool[new_size] = []
        if len(self._pool[new_size]) == 0:
            ptr = self._internal_cuda_allocate(new_size)
        else:
            ptr = self._pool[new_size].pop(-1)
        self._ptr_mem_size[ptr] = new_size
        return ptr

    def deallocate(self, ptr: driver.CUdeviceptr):
        size = self._ptr_mem_size.pop(ptr)
        if self._pool.get(size) is None:
            self._pool[size] = []
        self._pool[size].append(ptr)

    def empty_cache(self):
        for ls in self._pool.values():
            for ptr in ls:
                self._internal_cuda_deallocate(ptr)
        self._pool = {}

    def destroy(self):
        self.empty_cache()

    def _internal_cuda_allocate(self, size: int) -> driver.CUdeviceptr:
        return check_cuda_errors(driver.cuMemAlloc(size))

    def _internal_cuda_deallocate(self, ptr: driver.CUdeviceptr):
        check_cuda_errors(driver.cuMemFree(ptr))
