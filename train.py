import argparse
import json
import os
import pickle
import sys
import time
import subprocess
from pathlib import Path


def ensure_backend():
    project_root = Path(__file__).resolve().parent
    backend_src = project_root / "cpp_backend"
    build_dir = backend_src / "build"

    build_dir.mkdir(parents=True, exist_ok=True)

    # Search for compiled backend
    backend_module = None
    for file in build_dir.rglob("cpp_backend_ext*"):
        if file.suffix in {".pyd", ".so"}:
            backend_module = file
            break

    # If not found, build it
    if backend_module is None:
        print("C++ backend not found. Building now...")
        subprocess.run(
            ["cmake", "-S", str(backend_src), "-B", str(build_dir)],
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--config", "Release"],
            check=True,
        )

        for file in build_dir.rglob("cpp_backend_ext*"):
            if file.suffix in {".pyd", ".so"}:
                backend_module = file
                break

    if backend_module is None:
        raise RuntimeError("Failed to build cpp_backend_ext.")

    sys.path.insert(0, str(backend_module.parent))


# Call it before importing
ensure_backend()

from data.dataset import ImageFolderDataset
from data.dataloader import DataLoader
from models.cnn import SimpleCNN

import cpp_backend_ext as _C

CrossEntropyLoss = _C.CrossEntropyLoss
SGD = _C.SGD
Tensor = _C.Tensor


def save_checkpoint(model, path, config):
    weights = {}
    for i, p in enumerate(model.parameters()):
        weights[f"param_{i}"] = p.data

    payload = {
        "weights": weights,
        "config": {
            "num_classes": int(config["num_classes"]),
            "batch_size": int(config.get("batch_size", 32)),
        },
    }

    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(args):
    config = load_config(args.config_path)
    num_classes = None
    epochs = int(config.get("epochs", 50))
    batch_size = int(config.get("batch_size", 32))
    lr = float(config.get("lr", 0.05))

    dataset = ImageFolderDataset(args.train_path)
    print(f"Dataset loading time: {dataset.loading_time:.3f} seconds")

    num_classes = dataset.num_classes
    config["num_classes"] = num_classes
    dataset.sync_num_classes_to_config(args.config_path)
    print(f"Detected num_classes from dataset: {num_classes}")
    print(f"Updated config file with detected num_classes: {args.config_path}")

    loader = DataLoader(dataset, batch_size, True)

    model = SimpleCNN(num_classes)
    criterion = CrossEntropyLoss()
    optimizer = SGD(lr)

    stats = model.compute_stats(batch_size)
    print("Model Parameters:", stats.parameters)
    print("Model MACs:", stats.macs)
    print("Model FLOPs:", stats.flops)

    for epoch in range(epochs):
        print(">>> Epoch started:", epoch + 1)
        epoch_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()

        for images, labels in loader:
            batch_data = []
            for img in images:
                batch_data.extend(img.data)

            x = Tensor(batch_data, [len(images), 3, 32, 32], True)

            logits = model.forward(x)
            loss = criterion.forward(logits, labels)

            loss.backward()
            optimizer.step(model.parameters())
            optimizer.zero_grad(model.parameters())
            model.clear_forward_cache()

            epoch_loss += loss.data[0]

            for i in range(len(labels)):
                row = logits.data[i * num_classes:(i + 1) * num_classes]
                pred = row.index(max(row))
                if pred == labels[i]:
                    correct += 1
                total += 1

        epoch_time = time.time() - start_time
        acc = 100.0 * correct / total

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {epoch_loss:.4f} "
            f"Accuracy: {acc:.2f}% "
            f"Time: {epoch_time:.2f}s"
        )

    save_checkpoint(model, args.save_path, config)
    print("Training complete. Weights saved to:", args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--save_path", default="model_weights.pkl")
    parsed_args = parser.parse_args()

    main(parsed_args)
