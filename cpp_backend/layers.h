#ifndef LAYERS_H
#define LAYERS_H

#include "tensor.h"

class Linear {
public:
    Tensor weight;
    Tensor bias;

    Linear(int in_features, int out_features);
    Tensor forward(Tensor& x);
};

class ReLU {
public:
    Tensor forward(Tensor& x);
};

class Conv2D {
public:
    Tensor weight;
    Tensor bias;
    int in_channels, out_channels;
    int kernel_size;
    int stride, padding;

    Conv2D(int in_channels,
           int out_channels,
           int kernel_size,
           int stride = 1,
           int padding = 0);

    Tensor forward(Tensor& x);
};

class MaxPool2D {
public:
    int kernel_size;
    int stride;

    MaxPool2D(int kernel_size, int stride);
    Tensor forward(Tensor& x);
};

#endif // LAYERS_H
