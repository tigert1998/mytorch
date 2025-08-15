extern "C" __global__ void conv2d_reference(
    int input_h, int input_w, int in_channels, int out_channels,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int padding_h, int padding_w, float *input, float *weight, float *bias,
    float *output) {
  int output_h = (input_h + padding_h * 2 - kernel_size_h) / stride_h + 1;
  int output_w = (input_w + padding_w * 2 - kernel_size_w) / stride_w + 1;

  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  int yid = blockIdx.y * blockDim.y + threadIdx.y;
  int zid = blockIdx.z * blockDim.z + threadIdx.z;

  if (xid >= out_channels || yid >= output_h || zid >= output_w) return;
  float ans = 0;
  for (int i = 0; i < kernel_size_h; i++)
    for (int j = 0; j < kernel_size_w; j++)
      for (int k = 0; k < in_channels; k++) {
        float weight_value =
            weight[((xid * in_channels + k) * kernel_size_h + i) *
                       kernel_size_w +
                   j];
        int input_x = yid * stride_h + i - padding_h;
        int input_y = zid * stride_w + j - padding_w;
        bool is_padding = input_x < 0 || input_y < 0 || input_x >= input_h ||
                          input_y >= input_w;
        float input_value =
            is_padding ? 0 : input[(k * input_h + input_x) * input_w + input_y];
        float bias_value = bias != NULL ? bias[xid] : 0;
        ans += input_value * weight_value + bias_value;
      }

  output[(xid * output_h + yid) * output_w + zid] = ans;
}
