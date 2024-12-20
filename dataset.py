import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms


class FourMNISTBags(Dataset):
    def __init__(
            self,
            root='./data',
            split='train',
            num_bags=2500,
            mean_bag_size=30,
            std_bag_size=2,
            target_numbers=[8, 9],
            target_probability=0.1,
            seed=0):
        self.num_bags = num_bags
        self.X = []
        self.y = []

        dataset = datasets.MNIST(
            root=root,
            train=split=='train',
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        if split != 'train':
            dataset = random_split(
                dataset=dataset,
                lengths=[0.5, 0.5],
                generator=torch.Generator().manual_seed(42)
            )[1 if split == 'test' else 0]

        digit_indices = [[] for _ in range(3)]
        for i, (_, y) in enumerate(dataset):
            if y == target_numbers[0]:
                digit_indices[1].append(i)
            elif y == target_numbers[1]:
                digit_indices[2].append(i)
            else:
                digit_indices[0].append(i)
        for i in range(3):
            digit_indices[i] = np.array(digit_indices[i])

        rng = np.random.default_rng(seed)
        for i in range(self.num_bags):
            label = i % 4
            bag_size = max(2, np.round(rng.normal(loc=mean_bag_size, scale=std_bag_size)).astype(int))
            digit_counts = np.zeros(3, dtype=int)
            digit_counts[0] = bag_size
            if label == 3:
                target_count = max(2, (rng.random(bag_size) < target_probability).sum())
                digit_counts[0] -= target_count
                digit_counts[1] = np.clip((rng.random(target_count) < 0.5).sum(), 1, target_count - 1)
                digit_counts[2] = target_count - digit_counts[1]
            elif label != 0:
                target_count = max(1, (rng.random(bag_size) < target_probability).sum())
                digit_counts[0] -= target_count
                digit_counts[label] = target_count
            indices = np.concatenate([digit_indices[i][rng.integers(len(digit_indices[i]), size=digit_counts[i])] for i in [1, 0, 2]])
            self.X.append(torch.stack([dataset[i][0] for i in indices]))
            self.y.append(label)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.num_bags


if __name__ == '__main__':
    for split, num_bags, seed in zip(['train', 'valid', 'test'], [2500, 1000, 1000], [0, 1, 2]):
        dataset = FourMNISTBags(split=split, num_bags=num_bags, seed=seed)
        print(f'{split}\t: {len(dataset):4d}')
        counts = np.zeros(4, dtype=int)
        for _, y in dataset:
            counts[y] += 1
        for i in range(4):
            print(f'class {i}\t: {counts[i]:4d}')