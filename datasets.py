import torch
import torchvision
def load_Fashion_MNIST(batch_size_train, batch_size_test):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST('/mloraw1/hashemi/', train=True, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(
                                                  (0.5,), (0.5,))
                                          ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST('/mloraw1/hashemi/', train=False, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(
                                                  (0.5,), (0.5,))
                                          ])),
        batch_size=batch_size_test, shuffle=True)

    return train_loader, test_loader
def load_MNIST(batch_size_train, batch_size_test):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/mloraw1/hashemi/', train=True, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(
                                                  (0.1307,), (0.3081,))
                                          ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/mloraw1/hashemi/', train=False, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(
                                                  (0.1307,), (0.3081,))
                                          ])),
        batch_size=batch_size_test, shuffle=True)
    return train_loader, test_loader

