import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import wandb
from datasets import load_Fashion_MNIST, load_MNIST
import numpy as np
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.05
n_epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad = torch.zeros(p.grad.shape, device=device)
        return

    def get_gradients(self):
        gradients = []
        for p in self.parameters():
            gradients.append(p.grad)
        return gradients

    def update(self, gradients):
        for i, p in enumerate(self.parameters(), 0):
            p.data -= learning_rate*gradients[i]/2
        return



wandb.init(
    project="personalization",
    name = "2 FMNIST + FMNIST-F - agg = 1+F",
    # track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "epochs": n_epochs,
        "batch_size": batch_size_train
    }
)


def evaluate(model, test_dataset):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataset:
            output = model(data.to(device))
            test_loss += F.nll_loss(output, target.to(device), size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.to(device).data.view_as(pred)).sum()
    test_loss /= len(test_dataset.dataset)
    test_acc = 100. * correct / len(test_dataset.dataset)
    wandb.log({"accuracy": test_acc, "loss": test_loss})
    print(test_acc)

def centralized_train(model, test_dataset, epochs, aggregator, *datasets):
    for epoch in tqdm(range(epochs)):
        model.train()
        for batches in zip(*datasets):
            grads = list()
            # print(len(batches), len(batches[0]))
            for (data, target) in batches:
                model.zero_grad()
                output = model(data.to(device))
                loss = model.criterion(output, target.to(device))
                loss.backward()
                grads.append(model.get_gradients())
            aggregated_grads = aggregator(grads)
            model.update(aggregated_grads)

        evaluate(model, test_dataset)
    return


def decentralized_train(test_model, models, test_dataset, epochs, aggregator, datasets):
    for epoch in tqdm(range(epochs)):
        for i, model in enumerate(models):
            model.train()
            for (data, target) in datasets[i]:
                model.zero_grad()
                output = model(data.to(device))
                loss = model.criterion(output, target.to(device))
                loss.backward()
                model.update(model.get_gradients())

        average_models(models)
        evaluate(test_model, test_dataset)


def average_models(models):
    params = list()
    # print(type(params), len(params))
    for i in range(len(models)):
        for j, param in enumerate(models[i].parameters()):
            # print(type(param.data))
            # print(param.data.shape)
            if i == 0:
                params.append(param.data)
            elif i < 2 or j < 2:
                params[j] += param.data

    # print(len(params))
    for i in range(len(params)):
        # print(params[i].shape)
        if i < 2:
            params[i] /= 3
        else:
            params[i] /= 2

    for i in range(len(models)):
        model = models[i]
        for j, param in enumerate(model.parameters()):
            # if len(param.data.shape) == 4:
            if i + 1 < len(models) or j < 2:
                param.data = params[j]

    return
    # print(params)
    # exit(0)


def averaging(gradients):
    grads = []
    for i in range(len(gradients[0])):
        grads.append(gradients[0][i])
        for j in range(1, len(gradients)):
            grads[i] += gradients[j][i]

        grads[i]/= len(gradients)
    return grads


train_loader1, test_loader1 = load_Fashion_MNIST()
train_loader2, test_loader2 = load_Fashion_MNIST()
train_loader3, test_loader3 = load_Fashion_MNIST()

# print(train_loader3.dataset.targets)
train_loader3.dataset.targets = (train_loader2.dataset.targets + 5) % 10
# print(train_loader3.dataset.targets)


model1 = Net().to(device)
model2 = Net().to(device)
model3 = Net().to(device)

# print(device, type(train_loader1))

decentralized_train(model1, [model1, model2, model3], test_loader1, n_epochs, averaging, [train_loader1, train_loader2, train_loader3])