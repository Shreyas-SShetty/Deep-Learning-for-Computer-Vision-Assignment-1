#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

/*
 * CrossEntropyLoss for multiclass classification
 * Assumes:
 *  - logits shape: [N, num_classes]
 *  - targets shape: [N] (class indices)
 */
class CrossEntropyLoss {
public:
    Tensor forward(Tensor& logits, const std::vector<int>& targets);
};

#endif // LOSS_H
