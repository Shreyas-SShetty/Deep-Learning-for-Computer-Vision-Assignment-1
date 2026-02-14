import argparse
import json
import os
import pickle
import sys

from data.dataset import ImageFolderDataset
from data.dataloader import DataLoader
from models.cnn import SimpleCNN

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
backend_path = os.path.join(project_root, "build", "Release")

if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

import cpp_backend_ext as _C

Tensor = _C.Tensor


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_checkpoint(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and "weights" in payload:
        return payload["weights"], payload.get("config", {})

    # Backward compatibility with old checkpoints that only saved params.
    return payload, {}


def load_weights(model, weights):
    for i, p in enumerate(model.parameters()):
        p.data = weights[f"param_{i}"]


def resolve_runtime_config(args, saved_config):
    config = dict(saved_config)

    if args.config_path:
        config.update(load_config(args.config_path))

    if args.num_classes is not None:
        config["num_classes"] = args.num_classes

    return config


def main(args):
    weights, saved_config = read_checkpoint(args.weights_path)
    config = resolve_runtime_config(args, saved_config)

    if "num_classes" not in config:
        raise ValueError(
            "num_classes is required. Provide --num_classes/--config_path or use a new checkpoint from updated train.py."
        )

    num_classes = int(config["num_classes"])
    batch_size = int(config.get("batch_size", args.batch_size))

    dataset = ImageFolderDataset(args.test_path)
    print(f"Dataset loading time: {dataset.loading_time:.3f} seconds")

    loader = DataLoader(dataset, batch_size, False)

    model = SimpleCNN(num_classes)
    load_weights(model, weights)

    stats = model.compute_stats(batch_size)
    print("Model Parameters:", stats.parameters)
    print("Model MACs:", stats.macs)
    print("Model FLOPs:", stats.flops)

    correct = 0
    total = 0

    for images, labels in loader:
        batch_data = []
        for img in images:
            batch_data.extend(img.data)

        x = Tensor(batch_data, [len(images), 3, 32, 32], False)

        logits = model.forward(x)

        for i in range(len(labels)):
            row = logits.data[i * num_classes:(i + 1) * num_classes]
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
    parser.add_argument("--config_path")
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--batch_size", type=int, default=32)

    parsed_args = parser.parse_args()
    main(parsed_args)
