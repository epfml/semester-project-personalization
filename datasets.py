import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

class ConvexDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Dataset:

    def __init__(self, name, batch_size_train, batch_size_test, flipped=False, seed = 0):
        """
        initializing a dataset
        name: name of the dataset (we call its function with this name): fashion_MNIST, MNIST
        flipped: if the labels should be flipped
        seed: we shift the labels by this seed, e.g. if seed = 1 then all the labels increase by 1 except for 9 which becomes 0
        """

        self.name = name
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.flipped = flipped
        self.seed = seed
        dataset_to_use = getattr(self, self.name)
        self.train_loader, self.val_loader, self.test_loader = dataset_to_use()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def flip_labels(self, train_loader, test_loader):
        """flips the labels of the dataset by using seed"""
        train_loader.targets = (train_loader.targets + self.seed) % 10
        test_loader.dataset.targets =  (test_loader.dataset.targets + self.seed) % 10
        # print(train_loader.dataset.targets)
        return train_loader, test_loader

    def split_train(self, train_data):
        val_size = int(0.2*len(train_data))
        # val_size = 1
        train_size = len(train_data) - val_size
        train_data, val_data = random_split(train_data, [train_size, val_size])
        train_loader = DataLoader(train_data, batch_size=self.batch_size_train, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size_test, shuffle=True)
        return train_loader, val_loader


    def fashion_MNIST(self):
        """loads fashion MNIST"""
        train_data = torchvision.datasets.FashionMNIST('/mloraw1/hashemi/', train=True, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(
                                                      (0.5,), (0.5,))
                                              ]))


        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST('/mloraw1/hashemi/', train=False, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(
                                                      (0.5,), (0.5,))
                                              ])),
            batch_size=self.batch_size_test, shuffle=True)

        # print(type(train_data.dataset), type(test_loader.dataset))

        if self.flipped:
            train_loader, test_loader = self.flip_labels(train_data, test_loader)

        train_loader, val_loader = self.split_train(train_data)
        return train_loader, val_loader, test_loader


    def MNIST(self):
        """loads MNIST"""
        train_data = torchvision.datasets.MNIST('/mloraw1/hashemi/', train=True, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(
                                                      (0.1307,), (0.3081,))
                                              ]))

        test_loader = DataLoader(
            torchvision.datasets.MNIST('/mloraw1/hashemi/', train=False, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(
                                                      (0.1307,), (0.3081,))
                                              ])),
            batch_size= self.batch_size_test, shuffle=True)

        if self.flipped:
            train_loader, test_loader = self.flip_labels(train_data, test_loader)
        train_loader, val_loader = self.split_train(train_data)
        return train_loader, val_loader, test_loader

    # def convex_dataset(self, minimas, batch_size_train, batch_size_test):
    #     np.random.seed(0)
    #     X = 1000*np.random.rand(60000, len(minimas)) - 500
    #     y = list()
    #     for params in X:
    #         label = 0
    #         for i in range(len(params)):
    #             label += 1/2 * (params[i] - minimas[i]) ** 2
    #         y.append(label)
    #
    #     X = torch.tensor(X).to(self.device)
    #     y = torch.tensor(y).to(self.device)
    #
    #     train_dataset = ConvexDataset(X[:50000], y[:50000])
    #     test_dataset = ConvexDataset(X[50000:], y[50000:])
    #
    #     train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    #     test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)
    #
    #     return train_loader, test_loader

