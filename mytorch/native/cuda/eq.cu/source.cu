#include <cuda_fp16.h>

#include <cuda/std/cstdint>

#include "broadcast_utils.cuh"

template <typename T>
__global__ void eq_reference(int n, int x_shape_n, int* x_shape, int y_shape_n,
                             int* y_shape, T* x, T* y, int8_t* output) {
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  if (xid >= n) return;
  int2 pair = broadcast(xid, x_shape_n, x_shape, y_shape_n, y_shape);
  output[xid] = (int8_t)(x[pair.x] == y[pair.y]);
}
