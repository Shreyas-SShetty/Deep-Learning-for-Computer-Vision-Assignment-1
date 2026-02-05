

#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <functional>

class Tensor {
public:
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<int> shape;

    bool requires_grad = false;

    std::function<void(Tensor&)> backward_fn;
    std::vector<Tensor*> parents;

    Tensor();
    Tensor(const std::vector<float>& data_,
           const std::vector<int>& shape_,
           bool requires_grad_ = false);

    void zero_grad();
    void backward();
};

#endif

