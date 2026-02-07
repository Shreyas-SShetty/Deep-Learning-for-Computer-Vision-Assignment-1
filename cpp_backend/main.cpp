#include <iostream>
#include "tensor.h"
#include "ops.h"

int main() {

    // Create x and y
    Tensor x({2.0f}, {1}, true);
    Tensor y({3.0f}, {1}, true);

    // Forward: z = x^2 * y
    Tensor a = mul(x, x);   // x^2
    Tensor z = mul(a, y);   // x^2 * y

    std::cout << "z = " << z.data[0] << std::endl;

    // Backward
    z.backward();

    // Print gradients
    std::cout << "dz/dx = " << x.grad[0] << std::endl;
    std::cout << "dz/dy = " << y.grad[0] << std::endl;

    return 0;
}
