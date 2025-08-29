inline __device__ int restore_reduction(int shape_n, int* shape,
                                        int num_reduce_axis, int* reduce_axis,
                                        int outer_idx, int inner_idx) {
  int mul = 1, dest = 0;

  for (int i = shape_n - 1, j = num_reduce_axis - 1; i >= 0; i--) {
    if (j >= 0 && i == reduce_axis[j]) {
      dest += (inner_idx % shape[i]) * mul;
      inner_idx /= shape[i];
      j--;
    } else {
      dest += (outer_idx % shape[i]) * mul;
      outer_idx /= shape[i];
    }
    mul *= shape[i];
  }

  return dest;
}

template <typename T, typename Adapter, typename... Args>
__global__ void ReduceTemplate(Args... args) {
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  const int num_warps = blockDim.x / warpSize;

  extern __shared__ char shared[];
  Adapter adapter(shared, args...);

  int outer_size = adapter.outer();
  int inner_size = adapter.inner();

  for (int x = blockIdx.y; x < outer_size; x += gridDim.y) {
    adapter.InitOuter();
    for (int i = warp_id * warpSize + lane_id; i < inner_size;
         i += num_warps * warpSize) {
      adapter.LoopInner(x, i);
    }
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      adapter.Aggregate(offset);
    }
    if (lane_id == 0) {
      adapter.WriteBuffer(warp_id);
    }
    __syncthreads();
    if (warp_id == 0) {
      adapter.ReadBuffer(lane_id);
#pragma unroll
      for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        adapter.Aggregate(offset);
      }
      if (lane_id == 0) {
        adapter.WriteAnswer(x);
      }
    }
  }
}