import numpy as np

from cuda.bindings import driver


class SimpleCudaMemoryAllocator:
    def __init__(self):
        self._pool = {}
        self._ptr_mem_size = {}

    def allocate(self, size):
        assert size >= 1
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

    def deallocate(self, ptr):
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

    def _internal_cuda_allocate(self, size):
        from mytorch.cuda.env import check_cuda_errors

        return check_cuda_errors(driver.cuMemAlloc(size))

    def _internal_cuda_deallocate(self, ptr):
        from mytorch.cuda.env import check_cuda_errors

        check_cuda_errors(driver.cuMemFree(ptr))
