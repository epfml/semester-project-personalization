import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ConvexDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Dataset:

    def __init__(self, name, batch_size_train, batch_size_test, flipped=False):
        print('flipped? ', flipped)
        self.name = name
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.flipped = flipped
        dataset_to_use = getattr(self, self.name)
        self.train_loader, self.test_loader = dataset_to_use()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def flip_labels(self, train_loader, test_loader):
        print('miaaam')
        seed = np.random.randint(1, 9)
        train_loader.dataset.targets = (train_loader.dataset.targets + seed) % 10
        test_loader.dataset.targets =  (test_loader.dataset.targets + seed) % 10
        print(train_loader.dataset.targets)
        return train_loader, test_loader
    def fashion_MNIST(self):
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST('/mloraw1/hashemi/', train=True, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(
                                                      (0.5,), (0.5,))
                                              ])),
            batch_size=self.batch_size_train, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST('/mloraw1/hashemi/', train=False, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(
                                                      (0.5,), (0.5,))
                                              ])),
            batch_size=self.batch_size_test, shuffle=True)

        if self.flipped:
            print("Flipping labels")
            train_loader, test_loader = self.flip_labels(train_loader, test_loader)

        return train_loader, test_loader


    def MNIST(self):
        train_loader = DataLoader(
            torchvision.datasets.MNIST('/mloraw1/hashemi/', train=True, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(
                                                      (0.1307,), (0.3081,))
                                              ])),
            batch_size= self.batch_size_train, shuffle=True)

        test_loader = DataLoader(
            torchvision.datasets.MNIST('/mloraw1/hashemi/', train=False, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(
                                                      (0.1307,), (0.3081,))
                                              ])),
            batch_size= self.batch_size_test, shuffle=True)

        if self.flipped:
            train_loader, test_loader = self.flip_labels(train_loader, test_loader)

        return train_loader, test_loader

    def convex_dataset(self, minimas, batch_size_train, batch_size_test):
        np.random.seed(0)
        X = 1000*np.random.rand(60000, len(minimas)) - 500
        y = list()
        for params in X:
            label = 0
            for i in range(len(params)):
                label += 1/2 * (params[i] - minimas[i]) ** 2
            y.append(label)

        X = torch.tensor(X).to(self.device)
        y = torch.tensor(y).to(self.device)

        train_dataset = ConvexDataset(X[:50000], y[:50000])
        test_dataset = ConvexDataset(X[50000:], y[50000:])

        train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)

        return train_loader, test_loader

