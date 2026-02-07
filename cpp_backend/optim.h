#ifndef OPTIM_H
#define OPTIM_H

#include "tensor.h"
#include <vector>

class SGD {
public:
    float lr;

    explicit SGD(float learning_rate);
    void step(const std::vector<Tensor*>& params);
    void zero_grad(const std::vector<Tensor*>& params);
};

#endif // OPTIM_H
