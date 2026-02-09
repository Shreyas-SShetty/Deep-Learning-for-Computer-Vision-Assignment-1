import random

class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)

        for start in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[start:start + self.batch_size]

            batch_images = []
            batch_labels = []

            for idx in batch_indices:
                img, label = self.dataset[idx]
                batch_images.append(img)
                batch_labels.append(label)

            yield batch_images, batch_labels

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
