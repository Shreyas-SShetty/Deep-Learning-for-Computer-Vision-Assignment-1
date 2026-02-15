import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time


def ensure_cpp_backend_built():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_backend_dir = os.path.join(current_dir, "cpp_backend")
    build_dir = os.path.join(cpp_backend_dir, "build")

    if os.path.isdir(build_dir):
        shutil.rmtree(build_dir)

    os.makedirs(build_dir, exist_ok=True)

    cmake_config_cmd = ["cmake", ".."]
    if os.name == "nt":
        cmake_config_cmd.extend(["-A", "x64"])

    subprocess.run(cmake_config_cmd, cwd=build_dir, check=True)
    subprocess.run(["cmake", "--build", ".", "--config", "Release"], cwd=build_dir, check=True)


def import_runtime_modules():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_path = os.path.join(current_dir, "cpp_backend", "build", "Release")

    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from data.dataset import ImageFolderDataset
    from data.dataloader import DataLoader
    from models.cnn import SimpleCNN
    import cpp_backend_ext as _C

    return ImageFolderDataset, DataLoader, SimpleCNN, _C


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
    ensure_cpp_backend_built()
    ImageFolderDataset, DataLoader, SimpleCNN, _C = import_runtime_modules()

    CrossEntropyLoss = _C.CrossEntropyLoss
    SGD = _C.SGD
    Tensor = _C.Tensor

    config = load_config(args.config_path)
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
