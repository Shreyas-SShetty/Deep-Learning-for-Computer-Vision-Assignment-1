#include "optim.h"

SGD::SGD(float learning_rate)
    : lr(learning_rate) {}

/*
 * Update parameters
 * param = param - lr * grad
 */
void SGD::step(const std::vector<Tensor*>& params) {
    for (Tensor* p : params) {
        if (!p || !p->requires_grad) continue;

        for (size_t i = 0; i < p->data.size(); i++) {
            p->data[i] -= lr * p->grad[i];
        }
    }
}

/*
 * Reset gradients after update
 */
void SGD::zero_grad(const std::vector<Tensor*>& params) {
    for (Tensor* p : params) {
        if (!p || !p->requires_grad) continue;
        p->zero_grad();
    }
}
