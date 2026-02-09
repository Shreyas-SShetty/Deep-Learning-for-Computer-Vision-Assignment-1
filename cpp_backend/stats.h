#ifndef STATS_H
#define STATS_H

#include <vector>

/*
 * Simple struct to hold model statistics
 */
struct ModelStats {
    long long parameters;
    long long macs;
    long long flops;
};

/* ---- Layer-wise stats ---- */
long long linear_params(int in_features, int out_features);
long long linear_macs(int batch_size, int in_features, int out_features);

long long conv2d_params(int in_channels, int out_channels, int kernel_size);
long long conv2d_macs(int batch_size,
                      int out_channels,
                      int out_h,
                      int out_w,
                      int in_channels,
                      int kernel_size);

/* ---- Utility ---- */
ModelStats finalize_stats(long long params, long long macs);

#endif // STATS_H
