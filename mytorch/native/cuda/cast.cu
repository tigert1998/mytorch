#include <cuda_fp16.h>

#include <cuda/std/cstdint>

template <typename T1, typename T2>
__global__ void cast_reference(int n, T1* input, T2* output) {
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize + blockIdx.x * blockDim.x / warpSize;
  int num_warps = gridDim.x * blockDim.x / warpSize;
  for (int i = warp_id * warpSize; i < n; i += num_warps * warpSize) {
    int xid = i + lane_id;
    if (xid < n) output[xid] = (T2)input[xid];
  }
}