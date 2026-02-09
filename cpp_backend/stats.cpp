#include "stats.h"

/*
 * Parameters:
 *  weights = in_features * out_features
 *  bias    = out_features
 */
long long linear_params(int in_features, int out_features) {
    return (long long)in_features * out_features + out_features;
}

/*
 * MACs:
 *  For each output neuron:
 *    in_features MACs
 */
long long linear_macs(int batch_size,
                      int in_features,
                      int out_features) {
    return (long long)batch_size *
           in_features *
           out_features;
}

/*
 * Parameters:
 *  weights = out_channels * in_channels * k * k
 *  bias    = out_channels
 */
long long conv2d_params(int in_channels,
                        int out_channels,
                        int kernel_size) {
    return (long long)out_channels *
           in_channels *
           kernel_size *
           kernel_size
           + out_channels;
}

/*
 * MACs:
 *  For each output pixel:
 *    in_channels * k * k MACs
 */
long long conv2d_macs(int batch_size,
                      int out_channels,
                      int out_h,
                      int out_w,
                      int in_channels,
                      int kernel_size) {
    return (long long)batch_size *
           out_channels *
           out_h *
           out_w *
           in_channels *
           kernel_size *
           kernel_size;
}

ModelStats finalize_stats(long long params,
                          long long macs) {
    ModelStats s;
    s.parameters = params;
    s.macs = macs;
    s.flops = 2 * macs;  // standard DL convention
    return s;
}
