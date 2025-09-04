#include <cuda_fp16.h>

#include <cuda/std/cstdint>

__device__ int permute_shape(int xid, int shape_n, int* shape, int* permute) {
  int to = 0, tmp = xid;
  for (int i = shape_n - 1; i >= 0; i--) {
    int dim = tmp % shape[i];
    tmp /= shape[i];

    int idx;
    for (idx = 0; idx < shape_n; idx++)
      if (permute[idx] == i) break;

    int mul = 1;
    for (int j = shape_n - 1; j > idx; j--) mul = mul * shape[permute[j]];
    to += dim * mul;
  }
  return to;
}

template <typename T>
__global__ void permute_reference(int n, T* input, T* output, int shape_n,
                                  int* shape, int* permute) {
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  if (xid >= n) return;

  int to = permute_shape(xid, shape_n, shape, permute);
  output[to] = input[xid];
}

template <typename T>
__global__ void permute_backward_reference(int n, int shape_n, int* shape,
                                           int* permute, T* input_grad,
                                           T* output_grad) {
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  if (xid >= n) return;

  int to = permute_shape(xid, shape_n, shape, permute);
  input_grad[xid] = output_grad[to];
}
