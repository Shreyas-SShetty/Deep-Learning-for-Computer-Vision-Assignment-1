#include "ops.h"
#include <cassert>
#include <numeric>

Tensor add(Tensor& a, Tensor& b) {
    Tensor out;
    out.requires_grad = a.requires_grad || b.requires_grad;
    out.parents = { &a, &b };

    // ---- shape checks ----
    int a_dims = a.shape.size();
    int b_dims = b.shape.size();

    assert(a_dims >= 1);
    int C = a.shape.back();

    bool bias_case = false;

    // b: [C]
    if (b_dims == 1 && b.shape[0] == C) {
        bias_case = true;
    }

    // b: [1, C]
    else if (b_dims == 2 && b.shape[0] == 1 && b.shape[1] == C) {
        bias_case = true;
    }
    // normal elementwise
    else {
        assert(a.shape == b.shape);
    }

    // ---- forward ----
    out.shape = a.shape;
    out.data.resize(a.data.size());

    int N = a.data.size() / C;

    if (!bias_case) {
        // elementwise add
        for (size_t i = 0; i < a.data.size(); i++) {
            out.data[i] = a.data[i] + b.data[i];
        }
    } else {
        // broadcast bias
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < C; j++) {
                out.data[i * C + j] =
                    a.data[i * C + j] + b.data[j];
            }
        }
    }

    if (out.requires_grad) {
        out.parents = {&a, &b};
        out.backward_fn = [&, C, N, bias_case](Tensor& self) {
            // dL/da
            if (a.requires_grad) {
                for (size_t i = 0; i < a.grad.size(); i++) {
                    a.grad[i] += self.grad[i];
                }
            }

            // dL/db
            if (b.requires_grad) {
                if (!bias_case) {
                    for (size_t i = 0; i < b.grad.size(); i++) {
                        b.grad[i] += self.grad[i];
                    }
                } else {
                    for (int j = 0; j < C; j++) {
                        float g = 0.0f;
                        for (int i = 0; i < N; i++) {
                            g += self.grad[i * C + j];
                        }
                        b.grad[j] += g;
                    }
                }
            }
        };
    }

    return out;
}

Tensor mul(Tensor& a, Tensor& b) {
    assert(a.data.size() == b.data.size());

    std::vector<float> out_data(a.data.size());
    for (size_t i = 0; i < a.data.size(); i++)
        out_data[i] = a.data[i] * b.data[i];

    Tensor out(out_data, a.shape, a.requires_grad || b.requires_grad);

    if (out.requires_grad) {
        out.parents = {&a, &b};
        out.backward_fn = [&](Tensor& self) {
            for (size_t i = 0; i < self.grad.size(); i++) {
                if (a.requires_grad) a.grad[i] += b.data[i] * self.grad[i];
                if (b.requires_grad) b.grad[i] += a.data[i] * self.grad[i];
            }
        };
    }

    return out;
}

Tensor matmul(Tensor& a, Tensor& b) {
    assert(a.shape.size() == 2 && b.shape.size() == 2);
    assert(a.shape[1] == b.shape[0]);

    int M = a.shape[0], K = a.shape[1], N = b.shape[1];
    std::vector<float> out_data(M * N, 0.0f);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < K; k++)
                out_data[i*N + j] += a.data[i*K + k] * b.data[k*N + j];

    Tensor out(out_data, {M, N}, a.requires_grad || b.requires_grad);

    if (out.requires_grad) {
        out.parents = {&a, &b};
        out.backward_fn = [&](Tensor& self) {
            if (a.requires_grad) {
                for (int i = 0; i < M; i++)
                    for (int k = 0; k < K; k++) {
                        float g = 0.0f;
                        for (int j = 0; j < N; j++)
                            g += self.grad[i*N + j] * b.data[k*N + j];
                        a.grad[i*K + k] += g;
                    }
            }
            if (b.requires_grad) {
                for (int k = 0; k < K; k++)
                    for (int j = 0; j < N; j++) {
                        float g = 0.0f;
                        for (int i = 0; i < M; i++)
                            g += a.data[i*K + k] * self.grad[i*N + j];
                        b.grad[k*N + j] += g;
                    }
            }
        };
    }

    return out;
}

Tensor sum(Tensor& a) {
    float s = std::accumulate(a.data.begin(), a.data.end(), 0.0f);
    Tensor out({s}, {1}, a.requires_grad);

    if (out.requires_grad) {
        out.parents = {&a};
        out.backward_fn = [&](Tensor& self) {
            for (size_t i = 0; i < a.grad.size(); i++)
                a.grad[i] += self.grad[0];
        };
    }

    return out;
}


Tensor mean(Tensor& a) {
    float s = std::accumulate(a.data.begin(), a.data.end(), 0.0f);
    float m = s / a.data.size();
    Tensor out({m}, {1}, a.requires_grad);

    if (out.requires_grad) {
        out.parents = {&a};
        out.backward_fn = [&](Tensor& self) {
            float g = self.grad[0] / a.data.size();
            for (size_t i = 0; i < a.grad.size(); i++)
                a.grad[i] += g;
        };
    }

    return out;
}


Tensor reshape(Tensor& a, const std::vector<int>& new_shape) {
    int new_size = 1;
    for (int s : new_shape) new_size *= s;
    assert(new_size == (int)a.data.size());

    Tensor out(a.data, new_shape, a.requires_grad);

    if (out.requires_grad) {
        out.parents = {&a};
        out.backward_fn = [&](Tensor& self) {
            for (size_t i = 0; i < a.grad.size(); i++)
                a.grad[i] += self.grad[i];
        };
    }

    return out;
}


Tensor flatten(Tensor& a) {
    return reshape(a, {(int)a.data.size()});
}
