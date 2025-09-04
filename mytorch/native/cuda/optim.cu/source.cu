#include <cuda_fp16.h>

template <typename T>
__global__ void sgd_reference(bool is_first_time, int numel, T* param, T* grad,
                              T* momentum_buffer, T lr, T weight_decay,
                              T momentum, T dampening, bool nesterov,
                              bool maximize) {
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  if (xid >= numel) return;

  T g = maximize ? -grad[xid] : grad[xid];
  g += weight_decay * param[xid];
  momentum_buffer[xid] =
      is_first_time
          ? g
          : (momentum * momentum_buffer[xid] + ((T)1 - dampening) * g);
  g = nesterov ? (g + momentum * momentum_buffer[xid]) : momentum_buffer[xid];
  param[xid] -= g * lr;
}
