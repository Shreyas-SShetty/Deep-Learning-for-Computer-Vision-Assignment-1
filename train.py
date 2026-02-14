import argparse
import time
import pickle
import os
import sys

from data.dataset import ImageFolderDataset
from data.dataloader import DataLoader
from models.cnn import SimpleCNN
#import sys
#sys.path.insert(
#    0,
#    r"C:\Users\shrey\Desktop\cminds\GNR638\Deep-Learning-for-Computer-Vision-Assignment-1\cpp_backend\build\Release"
#)

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.join(current_dir, "build", "Release")

if backend_path not in sys.path:
    sys.path.insert(0, backend_path)
import cpp_backend_ext as _C
CrossEntropyLoss = _C.CrossEntropyLoss
SGD = _C.SGD
Tensor = _C.Tensor

def save_weights(model, path):
    weights = {}
    for i, p in enumerate(model.parameters()):
        weights[f"param_{i}"] = p.data
    with open(path, "wb") as f:
        pickle.dump(weights, f)

def main(args):
    # -------- Dataset --------
    dataset = ImageFolderDataset(args.train_path)
    print(f"Dataset loading time: {dataset.loading_time:.3f} seconds")

    loader = DataLoader(dataset, args.batch_size, True)

    # -------- Model --------
    model = SimpleCNN(args.num_classes)
    criterion = CrossEntropyLoss()
    optimizer = SGD(args.lr)

    # -------- Stats --------
    stats = model.compute_stats(args.batch_size)
    print("Model Parameters:", stats.parameters)
    print("Model MACs:", stats.macs)
    print("Model FLOPs:", stats.flops)

    # -------- Training --------
    for epoch in range(args.epochs):
        print(">>> Epoch started:", epoch + 1)
        epoch_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()

        for images, labels in loader:
            # Stack batch manually
            batch_data = []
            for img in images:
                batch_data.extend(img.data)

            x = Tensor(batch_data, [len(images), 3, 32, 32], True)

            # Forward
            logits = model.forward(x)
            # print('logits', logits.data, logits.shape)
            loss = criterion.forward(logits, labels)
            # print('Loss:', loss.data[0])

            # Backward
            loss.backward()
            # print('logits grad:', logits.grad)
            optimizer.step(model.parameters())
            optimizer.zero_grad(model.parameters())
            model.clear_forward_cache()

            epoch_loss += loss.data[0]

            # Accuracy
            for i in range(len(labels)):
                row = logits.data[i * args.num_classes:
                                  (i + 1) * args.num_classes]
                pred = row.index(max(row))
                if pred == labels[i]:
                    correct += 1
                total += 1

        epoch_time = time.time() - start_time
        acc = 100.0 * correct / total

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Loss: {epoch_loss:.4f} "
              f"Accuracy: {acc:.2f}% "
              f"Time: {epoch_time:.2f}s")

    # -------- Save weights --------
    save_weights(model, args.save_path)
    print("Training complete. Weights saved to:", args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path")
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--save_path", default="model_weights.pkl")
    args = parser.parse_args()

    if args.train_path is None:
        args.train_path = input("Enter training dataset path: ")

    if args.num_classes is None:
        args.num_classes = int(input("Enter number of classes: "))
    
    args.epochs = 50       
    args.batch_size = 32   
    args.lr = 0.05        
    main(args)
