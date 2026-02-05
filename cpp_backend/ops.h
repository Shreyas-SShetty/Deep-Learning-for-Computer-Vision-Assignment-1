#ifndef OPS_H
#define OPS_H

#include "tensor.h"

Tensor add(Tensor& a, Tensor& b);
Tensor mul(Tensor& a, Tensor& b);
Tensor matmul(Tensor& a, Tensor& b);
Tensor sum(Tensor& a);
Tensor mean(Tensor& a);
Tensor reshape(Tensor& a, const std::vector<int>& new_shape);
Tensor flatten(Tensor& a);

#endif
