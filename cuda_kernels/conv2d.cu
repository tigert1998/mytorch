#include <cuda_fp16.h>

template <typename T>
__global__ void conv2d_reference(int batch_size, int input_h, int input_w,
                                 int in_channels, int out_channels,
                                 int kernel_size_h, int kernel_size_w,
                                 int stride_h, int stride_w, int padding_h,
                                 int padding_w, T *input, T *weight, T *bias,
                                 T *output) {
  int output_h = (input_h + padding_h * 2 - kernel_size_h) / stride_h + 1;
  int output_w = (input_w + padding_w * 2 - kernel_size_w) / stride_w + 1;

  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  int yid = blockIdx.y * blockDim.y + threadIdx.y;
  int zid = blockIdx.z * blockDim.z + threadIdx.z;

  if (xid >= batch_size * out_channels || yid >= output_h || zid >= output_w)
    return;
  int batch_idx = xid / out_channels;
  int out_c = xid % out_channels;

  T ans = 0;
  for (int i = 0; i < kernel_size_h; i++)
    for (int j = 0; j < kernel_size_w; j++)
      for (int k = 0; k < in_channels; k++) {
        T weight_value =
            weight[((out_c * in_channels + k) * kernel_size_h + i) *
                       kernel_size_w +
                   j];
        int input_x = yid * stride_h + i - padding_h;
        int input_y = zid * stride_w + j - padding_w;
        bool is_padding = input_x < 0 || input_y < 0 || input_x >= input_h ||
                          input_y >= input_w;
        T input_value =
            is_padding
                ? (T)0
                : input[((batch_idx * in_channels + k) * input_h + input_x) *
                            input_w +
                        input_y];
        T bias_value = bias != nullptr ? bias[out_c] : (T)0;
        ans += input_value * weight_value + bias_value;
      }

  output[(xid * output_h + yid) * output_w + zid] = ans;
}

extern "C" __global__ void conv2d_reference_fp32(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, float *input, float *weight, float *bias,
    float *output) {
  return conv2d_reference<float>(batch_size, input_h, input_w, in_channels,
                                 out_channels, kernel_size_h, kernel_size_w,
                                 stride_h, stride_w, padding_h, padding_w,
                                 input, weight, bias, output);
}

extern "C" __global__ void conv2d_reference_fp16(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, half *input, half *weight, half *bias,
    half *output) {
  return conv2d_reference<half>(batch_size, input_h, input_w, in_channels,
                                out_channels, kernel_size_h, kernel_size_w,
                                stride_h, stride_w, padding_h, padding_w, input,
                                weight, bias, output);
}
