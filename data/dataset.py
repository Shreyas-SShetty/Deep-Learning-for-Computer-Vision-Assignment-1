import os
import time
import cv2
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
print("project_root:", project_root)
backend_path = os.path.join(project_root, "cpp_backend", "build", "Release")
print("backend_path:", backend_path)

if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

import cpp_backend_ext as _C
Tensor = _C.Tensor

class ImageFolderDataset:
    def __init__(self, root_dir, image_size=32):
        """
        root_dir/
            class_0/
                img1.png
                img2.png
            class_1/
                img3.png
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.images = []
        self.labels = []
        self.class_to_idx = {}

        start_time = time.time()
        self._load_dataset()
        end_time = time.time()

        self.loading_time = end_time - start_time

    def _load_dataset(self):
        class_names = sorted(
            [d for d in os.listdir(self.root_dir)
             if os.path.isdir(os.path.join(self.root_dir, d))]
        )
        print("class_names:", class_names)

        # Assign numeric labels
        for idx, class_name in enumerate(class_names):
            self.class_to_idx[class_name] = idx

        print("class_to_idx:", self.class_to_idx)

        for class_name in class_names:
            class_path = os.path.join(self.root_dir, class_name)
            print("class_path:", class_path)
            label = self.class_to_idx[class_name]

            for fname in os.listdir(class_path):
                if not fname.lower().endswith(".png"):
                    continue

                img_path = os.path.join(class_path, fname)

                # OpenCV image loading
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                img = cv2.resize(img,
                                 (self.image_size, self.image_size))

                # Convert to CHW and normalize
                img = img.astype("float32") / 255.0
                img = img.transpose(2, 0, 1)  # C, H, W

                self.images.append(img)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # Flatten image for Tensor constructor
        tensor = Tensor(img.flatten().tolist(), list(img.shape), False)

        return tensor, label
