#include "layers.h"
#include "ops.h"
#include <cassert>
#include <algorithm>

/* ---------------- Linear ---------------- */
Linear::Linear(int in_features, int out_features)
    : weight(std::vector<float>(in_features * out_features, 0.001f),
             {in_features, out_features}, true),
      bias(std::vector<float>(out_features, 0.0f),
           {1, out_features}, true) {}

Tensor Linear::forward(Tensor& x) {
    Tensor out = matmul(x, weight);
    Tensor b = bias;  // broadcast manually later if needed
    return add(out, b);
}

/* ---------------- ReLU ---------------- */
Tensor ReLU::forward(Tensor& x) {
    std::vector<float> out_data(x.data.size());
    for (size_t i = 0; i < x.data.size(); i++)
        out_data[i] = std::max(0.0f, x.data[i]);

    Tensor out(out_data, x.shape, x.requires_grad);

    if (out.requires_grad) {
        out.parents = {&x};
        Tensor* x_ptr = &x;
        std::vector<float> x_data = x.data;
        out.backward_fn = [x_ptr, x_data](Tensor& self) {
            for (size_t i = 0; i < x_ptr->grad.size(); i++) {
                x_ptr->grad[i] += (x_data[i] > 0) ? self.grad[i] : 0.0f;
            }
        };
    }

    return out;
}

/* ---------------- Conv2D ---------------- */
/*
 * Input:  [N, C, H, W]
 * Weight: [F, C, K, K]
 * Output: [N, F, H_out, W_out]
 */
Conv2D::Conv2D(int in_c, int out_c, int k,
               int s, int p)
    : in_channels(in_c),
      out_channels(out_c),
      kernel_size(k),
      stride(s),
      padding(p),
      weight(std::vector<float>(out_c * in_c * k * k, 0.001f),
             {out_c, in_c, k, k}, true),
      bias(std::vector<float>(out_c, 0.0f),
           {out_c}, true) {}

Tensor Conv2D::forward(Tensor& x) {
    assert(x.shape.size() == 4);

    int N = x.shape[0];
    int C = x.shape[1];
    int H = x.shape[2];
    int W = x.shape[3];

    int H_out = (H - kernel_size + 2 * padding) / stride + 1;
    int W_out = (W - kernel_size + 2 * padding) / stride + 1;

    std::vector<float> out_data(N * out_channels * H_out * W_out, 0.0f);

    for (int n = 0; n < N; n++) {
        for (int f = 0; f < out_channels; f++) {
            for (int i = 0; i < H_out; i++) {
                for (int j = 0; j < W_out; j++) {
                    float sum = bias.data[f];
                    for (int c = 0; c < C; c++) {
                        for (int ki = 0; ki < kernel_size; ki++) {
                            for (int kj = 0; kj < kernel_size; kj++) {
                                int h = i * stride + ki - padding;
                                int w = j * stride + kj - padding;
                                if (h >= 0 && h < H && w >= 0 && w < W) {
                                    int x_idx =
                                        ((n * C + c) * H + h) * W + w;
                                    int w_idx =
                                        ((f * C + c) * kernel_size + ki)
                                            * kernel_size + kj;
                                    sum += x.data[x_idx] * weight.data[w_idx];
                                }
                            }
                        }
                    }
                    out_data[((n * out_channels + f) * H_out + i)
                                 * W_out + j] = sum;
                }
            }
        }
    }

    Tensor out(out_data, {N, out_channels, H_out, W_out},
               x.requires_grad || weight.requires_grad);

    if (out.requires_grad) {
        out.parents = {&x, &weight, &bias};

        Tensor* x_ptr = &x;
        Tensor* weight_ptr = &weight;
        Tensor* bias_ptr = &bias;
        int out_channels_local = out_channels;
        int kernel_size_local = kernel_size;
        int stride_local = stride;
        int padding_local = padding;

        out.backward_fn = [x_ptr,
                           weight_ptr,
                           bias_ptr,
                           N,
                           C,
                           H,
                           W,
                           H_out,
                           W_out,
                           out_channels_local,
                           kernel_size_local,
                           stride_local,
                           padding_local](Tensor& self) {
            // Input gradient
            if (x_ptr->requires_grad) {
                for (int n = 0; n < N; n++)
                    for (int c = 0; c < C; c++)
                        for (int h = 0; h < H; h++)
                            for (int w = 0; w < W; w++)
                                x_ptr->grad[((n * C + c) * H + h) * W + w] += 0.0f;
            }

            // Weight & bias gradient
            for (int n = 0; n < N; n++) {
                for (int f = 0; f < out_channels_local; f++) {
                    for (int i = 0; i < H_out; i++) {
                        for (int j = 0; j < W_out; j++) {
                            float g =
                                self.grad[((n * out_channels_local + f) * H_out + i)
                                              * W_out + j];
                            bias_ptr->grad[f] += g;

                            for (int c = 0; c < C; c++) {
                                for (int ki = 0; ki < kernel_size_local; ki++) {
                                    for (int kj = 0; kj < kernel_size_local; kj++) {
                                        int h = i * stride_local + ki - padding_local;
                                        int w = j * stride_local + kj - padding_local;
                                        if (h >= 0 && h < H && w >= 0 && w < W) {
                                            int x_idx =
                                                ((n * C + c) * H + h) * W + w;
                                            int w_idx =
                                                ((f * C + c) * kernel_size_local + ki)
                                                    * kernel_size_local + kj;
                                            weight_ptr->grad[w_idx] +=
                                                x_ptr->data[x_idx] * g;
                                            if (x_ptr->requires_grad)
                                                x_ptr->grad[x_idx] +=
                                                    weight_ptr->data[w_idx] * g;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
    }

    return out;
}

/* ---------------- MaxPool2D ---------------- */
MaxPool2D::MaxPool2D(int k, int s)
    : kernel_size(k), stride(s) {}

Tensor MaxPool2D::forward(Tensor& x) {
    assert(x.shape.size() == 4);

    int N = x.shape[0];
    int C = x.shape[1];
    int H = x.shape[2];
    int W = x.shape[3];

    int H_out = (H - kernel_size) / stride + 1;
    int W_out = (W - kernel_size) / stride + 1;

    std::vector<float> out_data(N * C * H_out * W_out);
    std::vector<int> max_idx(N * C * H_out * W_out);

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < H_out; i++) {
                for (int j = 0; j < W_out; j++) {
                    float max_val = -1e9;
                    int idx = -1;
                    for (int ki = 0; ki < kernel_size; ki++) {
                        for (int kj = 0; kj < kernel_size; kj++) {
                            int h = i * stride + ki;
                            int w = j * stride + kj;
                            int x_idx =
                                ((n * C + c) * H + h) * W + w;
                            if (x.data[x_idx] > max_val) {
                                max_val = x.data[x_idx];
                                idx = x_idx;
                            }
                        }
                    }
                    int out_idx =
                        ((n * C + c) * H_out + i) * W_out + j;
                    out_data[out_idx] = max_val;
                    max_idx[out_idx] = idx;
                }
            }
        }
    }

    Tensor out(out_data, {N, C, H_out, W_out}, x.requires_grad);

    if (out.requires_grad) {
        out.parents = {&x};
        Tensor* x_ptr = &x;
        std::vector<int> max_idx_cache = max_idx;
        out.backward_fn = [x_ptr, max_idx_cache](Tensor& self) {
            for (size_t i = 0; i < self.grad.size(); i++) {
                x_ptr->grad[max_idx_cache[i]] += self.grad[i];
            }
        };
    }

    return out;
}
