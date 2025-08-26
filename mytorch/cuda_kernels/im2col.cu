#include <cuda_fp16.h>

template <typename T>
__global__ void im2col_input_reference(int batch_size, int input_h, int input_w,
                                       int in_channels, int out_channels,
                                       int kernel_size_h, int kernel_size_w,
                                       int stride_h, int stride_w,
                                       int padding_h, int padding_w, T *input,
                                       T *bias, T *x) {
  int output_h = (input_h + padding_h * 2 - kernel_size_h) / stride_h + 1;
  int output_w = (input_w + padding_w * 2 - kernel_size_w) / stride_w + 1;

  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  int yid = blockIdx.y * blockDim.y + threadIdx.y;
  int zid = blockIdx.z * blockDim.z + threadIdx.z;

  if (xid >= batch_size || yid >= output_h || zid >= output_w) return;

  int x_rows = batch_size * output_h * output_w;
  int x_cols =
      kernel_size_h * kernel_size_w * in_channels + int(bias != nullptr);
  int x_cols_padded = (x_cols + 3) / 4 * 4;

  int x_row_offset = x_cols_padded * ((xid * output_h + yid) * output_w + zid);

  for (int k = 0; k < in_channels; k++)
    for (int i = 0; i < kernel_size_h; i++)
      for (int j = 0; j < kernel_size_w; j++) {
        int input_x = yid * stride_h + i - padding_h;
        int input_y = zid * stride_w + j - padding_w;
        bool is_padding = input_x < 0 || input_y < 0 || input_x >= input_h ||
                          input_y >= input_w;
        T input_value =
            is_padding ? (T)0
                       : input[((xid * in_channels + k) * input_h + input_x) *
                                   input_w +
                               input_y];

        int x_col_idx = (k * kernel_size_h + i) * kernel_size_w + j;
        x[x_row_offset + x_col_idx] = input_value;
      }

  if (bias != nullptr) {
    x[x_row_offset + x_cols - 1] = (T)1;
  }
  for (int i = x_cols; i < x_cols_padded; i++) {
    x[x_row_offset + i] = (T)0;
  }
}

extern "C" __global__ void im2col_input_reference_fp32(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, float *input, float *bias, float *x) {
  im2col_input_reference(batch_size, input_h, input_w, in_channels,
                         out_channels, kernel_size_h, kernel_size_w, stride_h,
                         stride_w, padding_h, padding_w, input, bias, x);
}

extern "C" __global__ void im2col_input_reference_fp16(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, half *input, half *bias, half *x) {
  im2col_input_reference(batch_size, input_h, input_w, in_channels,
                         out_channels, kernel_size_h, kernel_size_w, stride_h,
                         stride_w, padding_h, padding_w, input, bias, x);
}

template <typename T>
__global__ void im2col_weight_reference(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, T *weight, T *bias, T *y) {
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  int yid = blockIdx.y * blockDim.y + threadIdx.y;
  int zid = blockIdx.z * blockDim.z + threadIdx.z;

  if (xid >= out_channels * in_channels || yid >= kernel_size_h ||
      zid >= kernel_size_w)
    return;

  int out_c = xid / in_channels;
  int in_c = xid % in_channels;

  int y_rows = out_channels;
  int y_cols =
      kernel_size_h * kernel_size_w * in_channels + int(bias != nullptr);
  int y_cols_padded = (y_cols + 3) / 4 * 4;

  int y_row_offset = y_cols_padded * out_c;

  T weight_value = weight[((out_c * in_channels + in_c) * kernel_size_h + yid) *
                              kernel_size_w +
                          zid];

  int y_col_idx = (in_c * kernel_size_h + yid) * kernel_size_w + zid;
  y[y_row_offset + y_col_idx] = weight_value;

  if (y_col_idx == 0) {
    if (bias != nullptr) {
      y[y_row_offset + y_cols - 1] = bias[out_c];
    }
    for (int i = y_cols; i < y_cols_padded; i++) {
      y[y_row_offset + i] = (T)0;
    }
  }
}

extern "C" __global__ void im2col_weight_reference_fp32(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, float *weight, float *bias, float *y) {
  im2col_weight_reference(batch_size, input_h, input_w, in_channels,
                          out_channels, kernel_size_h, kernel_size_w, stride_h,
                          stride_w, padding_h, padding_w, weight, bias, y);
}

extern "C" __global__ void im2col_weight_reference_fp16(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, half *weight, half *bias, half *y) {
  im2col_weight_reference(batch_size, input_h, input_w, in_channels,
                          out_channels, kernel_size_h, kernel_size_w, stride_h,
                          stride_w, padding_h, padding_w, weight, bias, y);
}

template <typename T>
__global__ void reverse_im2col_input_reference(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, T *input, T *bias, T *x) {
  int output_h = (input_h + padding_h * 2 - kernel_size_h) / stride_h + 1;
  int output_w = (input_w + padding_w * 2 - kernel_size_w) / stride_w + 1;

  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  int yid = blockIdx.y * blockDim.y + threadIdx.y;
  int zid = blockIdx.z * blockDim.z + threadIdx.z;

  if (xid >= batch_size || yid >= output_h || zid >= output_w) return;

  int x_rows = batch_size * output_h * output_w;
  int x_cols =
      kernel_size_h * kernel_size_w * in_channels + int(bias != nullptr);
  int x_cols_padded = (x_cols + 3) / 4 * 4;

  int x_row_offset = x_cols_padded * ((xid * output_h + yid) * output_w + zid);

  for (int k = 0; k < in_channels; k++)
    for (int i = 0; i < kernel_size_h; i++)
      for (int j = 0; j < kernel_size_w; j++) {
        int input_x = yid * stride_h + i - padding_h;
        int input_y = zid * stride_w + j - padding_w;
        bool is_padding = input_x < 0 || input_y < 0 || input_x >= input_h ||
                          input_y >= input_w;
        if (!is_padding) {
          int x_col_idx = (k * kernel_size_h + i) * kernel_size_w + j;
          atomicAdd(
              &input[((xid * in_channels + k) * input_h + input_x) * input_w +
                     input_y],
              x[x_row_offset + x_col_idx]);
        }
      }
}

extern "C" __global__ void reverse_im2col_input_reference_fp32(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, float *input, float *bias, float *x) {
  reverse_im2col_input_reference(
      batch_size, input_h, input_w, in_channels, out_channels, kernel_size_h,
      kernel_size_w, stride_h, stride_w, padding_h, padding_w, input, bias, x);
}

extern "C" __global__ void reverse_im2col_input_reference_fp16(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, half *input, half *bias, half *x) {
  reverse_im2col_input_reference(
      batch_size, input_h, input_w, in_channels, out_channels, kernel_size_h,
      kernel_size_w, stride_h, stride_w, padding_h, padding_w, input, bias, x);
}

template <typename T>
__global__ void reverse_im2col_weight_reference(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, T *weight, T *bias, T *y) {
  int output_h = (input_h + padding_h * 2 - kernel_size_h) / stride_h + 1;
  int output_w = (input_w + padding_w * 2 - kernel_size_w) / stride_w + 1;

  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  int yid = blockIdx.y * blockDim.y + threadIdx.y;
  int zid = blockIdx.z * blockDim.z + threadIdx.z;

  if (xid >= out_channels * in_channels || yid >= kernel_size_h ||
      zid >= kernel_size_w)
    return;

  int out_c = xid / in_channels;
  int in_c = xid % in_channels;

  int y_rows = out_channels;
  int y_cols =
      kernel_size_h * kernel_size_w * in_channels + int(bias != nullptr);
  int y_cols_padded = (y_cols + 3) / 4 * 4;

  int y_row_offset = y_cols_padded * out_c;
  int y_col_idx = (in_c * kernel_size_h + yid) * kernel_size_w + zid;

  weight[((out_c * in_channels + in_c) * kernel_size_h + yid) * kernel_size_w +
         zid] = y[y_row_offset + y_col_idx];

  if (y_col_idx == 0 && bias != nullptr) {
    bias[out_c] = y[y_row_offset + y_cols - 1];
  }
}

extern "C" __global__ void reverse_im2col_weight_reference_fp32(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, float *weight, float *bias, float *y) {
  reverse_im2col_weight_reference(
      batch_size, input_h, input_w, in_channels, out_channels, kernel_size_h,
      kernel_size_w, stride_h, stride_w, padding_h, padding_w, weight, bias, y);
}

extern "C" __global__ void reverse_im2col_weight_reference_fp16(
    int batch_size, int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, half *weight, half *bias, half *y) {
  reverse_im2col_weight_reference(
      batch_size, input_h, input_w, in_channels, out_channels, kernel_size_h,
      kernel_size_w, stride_h, stride_w, padding_h, padding_w, weight, bias, y);
}