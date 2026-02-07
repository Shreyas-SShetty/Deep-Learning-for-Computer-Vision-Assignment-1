#include "loss.h"
#include <cmath>
#include <cassert>
#include <algorithm>

/*
 * Forward:
 *  - Applies softmax internally
 *  - Computes mean cross-entropy loss
 */
Tensor CrossEntropyLoss::forward(Tensor& logits,
                                 const std::vector<int>& targets) {
    assert(logits.shape.size() == 2);

    int N = logits.shape[0];
    int C = logits.shape[1];
    assert((int)targets.size() == N);

    std::vector<float> loss_data(1, 0.0f);
    std::vector<float> softmax(logits.data.size());

    // Softmax (numerically stable)
    for (int n = 0; n < N; n++) {
        float max_logit = -1e9;
        for (int c = 0; c < C; c++)
            max_logit = std::max(max_logit,
                                 logits.data[n * C + c]);

        float sum_exp = 0.0f;
        for (int c = 0; c < C; c++) {
            float e = std::exp(logits.data[n * C + c] - max_logit);
            softmax[n * C + c] = e;
            sum_exp += e;
        }

        for (int c = 0; c < C; c++)
            softmax[n * C + c] /= sum_exp;

        int y = targets[n];
        loss_data[0] -= std::log(softmax[n * C + y] + 1e-9f);
    }

    loss_data[0] /= N;

    Tensor loss(loss_data, {1}, logits.requires_grad);

    if (loss.requires_grad) {
        loss.parents = {&logits};

        loss.backward_fn = [&](Tensor& self) {
            for (int n = 0; n < N; n++) {
                for (int c = 0; c < C; c++) {
                    float grad = softmax[n * C + c];
                    if (c == targets[n])
                        grad -= 1.0f;
                    logits.grad[n * C + c] += grad / N;
                }
            }
        };
    }

    return loss;
}
