import argparse
import pickle

from data.dataset import ImageFolderDataset
from data.dataloader import DataLoader
from models.cnn import SimpleCNN

import sys
sys.path.insert(
    0,
    r"C:\Users\shrey\Desktop\cminds\GNR638\Deep-Learning-for-Computer-Vision-Assignment-1\cpp_backend\build\Release"
)
import cpp_backend_ext as _C
Tensor = _C.Tensor

def load_weights(model, path):
    with open(path, "rb") as f:
        weights = pickle.load(f)

    for i, p in enumerate(model.parameters()):
        p.data = weights[f"param_{i}"]


def main(args):
    # -------- Dataset --------
    dataset = ImageFolderDataset(args.test_path)
    print(f"Dataset loading time: {dataset.loading_time:.3f} seconds")

    loader = DataLoader(dataset, args.batch_size, False)

    # -------- Model --------
    model = SimpleCNN(args.num_classes)
    load_weights(model, args.weights_path)

    # -------- Stats --------
    stats = model.compute_stats(args.batch_size)
    print("Model Parameters:", stats.parameters)
    print("Model MACs:", stats.macs)
    print("Model FLOPs:", stats.flops)

    # -------- Evaluation --------
    correct = 0
    total = 0

    for images, labels in loader:
        batch_data = []
        for img in images:
            batch_data.extend(img.data)

        x = Tensor(batch_data, [len(images), 3, 32, 32], False)

        logits = model.forward(x)

        for i in range(len(labels)):
            row = logits.data[i * args.num_classes:
                              (i + 1) * args.num_classes]
            pred = row.index(max(row))
            if pred == labels[i]:
                correct += 1
            total += 1

    acc = 100.0 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    main(args)
