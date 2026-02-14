import sys
sys.path.insert(
    0,
    r"C:\Users\shrey\Desktop\cminds\GNR638\Deep-Learning-for-Computer-Vision-Assignment-1\cpp_backend\build\Release"
)

import cpp_backend_ext as _C

Conv2D = _C.Conv2D
ReLU = _C.ReLU
MaxPool2D = _C.MaxPool2D
Linear = _C.Linear
Tensor = _C.Tensor

conv2d_params = _C.conv2d_params
conv2d_macs = _C.conv2d_macs
linear_params = _C.linear_params
linear_macs = _C.linear_macs

finalize_stats = _C.finalize_stats

class SimpleCNN:
    def __init__(self, num_classes, input_channels=3):
        # ---- Layers ----
        self.conv = Conv2D(input_channels, 8, 3, 1, 0)

        self.relu = ReLU()
        self.pool = MaxPool2D(2, 2)

        # After conv + pool:
        # Input: 32x32
        # Conv (3x3): 30x30
        # Pool (2x2): 15x15
        self.fc_in_features = 8 * 15 * 15
        self.fc = Linear(self.fc_in_features, num_classes)

        # Keep references to forward-pass tensors so C++ parent pointers
        # remain valid until backward is executed.
        self._forward_cache = []

    # ---------------- Forward ----------------
    def forward(self, x):
        x0 = x
        x1 = self.conv.forward(x0)
        x2 = self.relu.forward(x1)
        x3 = self.pool.forward(x2)

        # Flatten
        # try to create reshape function in cpp-backend
        x3.shape = [x3.shape[0], self.fc_in_features]

        out = self.fc.forward(x3)

        # Preserve tensor lifetimes for autograd graph traversal.
        self._forward_cache = [x0, x1, x2, x3, out]

        return out

    # ---------------- Parameters ----------------
    def parameters(self):
        return [
            self.conv.weight,
            self.conv.bias,
            self.fc.weight,
            self.fc.bias
        ]

    def clear_forward_cache(self):
        self._forward_cache = []

    # ---------------- Model Stats ----------------
    def compute_stats(self, batch_size):
        params = 0
        macs = 0

        # Conv stats
        params += conv2d_params(3, 8, 3)

        macs += conv2d_macs(batch_size, 8, 30, 30, 3, 3)

        # FC stats
        params += linear_params(
            self.fc_in_features,
            self.fc.weight.shape[1]
        )

        macs += linear_macs(
            batch_size,
            self.fc_in_features,
            self.fc.weight.shape[1]
        )

        return finalize_stats(params, macs)
