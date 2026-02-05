#include "tensor.h"
#include <cassert>
#include <unordered_set>
#include <algorithm>

Tensor::Tensor() : requires_grad(false) {}


Tensor::Tensor(const std::vector<float>& data_,
               const std::vector<int>& shape_,
               bool requires_grad_)
    : data(data_), shape(shape_), requires_grad(requires_grad_) {
    if (requires_grad) {
        grad.resize(data.size(), 0.0f);
    }
}


void Tensor::zero_grad() {
    if (!requires_grad) return;
    std::fill(grad.begin(), grad.end(), 0.0f);
}


static void build_topo(Tensor* t,
                       std::unordered_set<Tensor*>& visited,
                       std::vector<Tensor*>& topo) {
    if (visited.count(t)) return;
    visited.insert(t);

    for (Tensor* p : t->parents) {
        build_topo(p, visited, topo);
    }
    topo.push_back(t);
}


void Tensor::backward() {
    assert(data.size() == 1 && "backward() only supports scalar loss");

    std::vector<Tensor*> topo;
    std::unordered_set<Tensor*> visited;

    build_topo(this, visited, topo);

    
    for (Tensor* t : topo) {
        if (t->requires_grad)
            std::fill(t->grad.begin(), t->grad.end(), 0.0f);
    }
    
    grad[0] = 1.0f;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Tensor* t = *it;
        if (t->backward_fn)
            t->backward_fn(*t);
    }
}
