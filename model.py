import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Net(nn.Module):
    def __init__(self):
        """initializing a simple NN with 2 convolutional layers and 2 fully connected layers"""
        super(Net, self).__init__()
        torch.autograd.set_detect_anomaly(True)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def forward(self, x):
        """rewriting the forward pass"""
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def zero_grad(self):
        """set gradients of the model to zero"""
        for p in self.parameters():
            if p.grad is not None:
                p.grad = torch.zeros(p.grad.shape, device=self.device)
        return

    def get_gradients(self):
        """return the gradients of the model"""
        gradients = []
        for i, p in enumerate(self.parameters()):
            gradients.append(p.grad)
        return gradients

    def update(self, gradients, learning_rate=0.1):
        """update the model parameters with the gradients and the given learning rate"""
        for i, p in enumerate(self.parameters()):
            if gradients[i] is not None:
                p.data -= learning_rate*gradients[i]

        return
