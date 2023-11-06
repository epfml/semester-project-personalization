import torch
import torchvision
import numpy as np
class Dataset:

    def __init__(self, dataset, batch_size_train, batch_size_test, flipped=False):
        self.dataset = dataset
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.flipped = flipped
        dataset_to_use = getattr(self, self.dataset)
        self.train_loader, self.test_loader = dataset_to_use()

    def flip_labels(self, train_loader, test_loader):

        seed = np.random.randint(1, 9)
        train_loader.dataset.targets = (train_loader.dataset.targets + seed) % 10
        test_loader.dataset.targets =  (test_loader.dataset.targets + seed) % 10
        print(train_loader.dataset.targets)
        return train_loader, test_loader
    def load_Fashion_MNIST(self):
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


    def load_MNIST(self):
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('/mloraw1/hashemi/', train=True, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(
                                                      (0.1307,), (0.3081,))
                                              ])),
            batch_size= self.batch_size_train, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
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

