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
            float z = logits.data[n*C + c] - max_logit;
            z = std::max(-20.0f, std::min(20.0f, z));
            softmax[n*C + c] = exp(z);
            sum_exp += softmax[n*C + c];
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

        Tensor* logits_ptr = &logits;
        std::vector<float> softmax_cache = softmax;
        std::vector<int> target_cache = targets;

        loss.backward_fn = [logits_ptr,
                            N,
                            C,
                            softmax_cache,
                            target_cache](Tensor& self) {
            for (int n = 0; n < N; n++) {
                for (int c = 0; c < C; c++) {
                    float grad = softmax_cache[n * C + c];
                    if (c == target_cache[n])
                        grad -= 1.0f;
                    logits_ptr->grad[n * C + c] += grad / N;
                }
            }
        };
    }

    return loss;
}
