import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Net(nn.Module):
    def __init__(self):
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
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        if x is not None and torch.isnan(x).any():
            print('Nan in cl1')
            breakpoint()
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        if x is not None and torch.isnan(x).any():
            print('Nan in cl2')
            breakpoint()
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        if x is not None and torch.isnan(x).any():
            print('Nan in fc1')
            breakpoint()
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        if x is not None and torch.isnan(x).any():
            print('Nan in fc2')
            breakpoint()
        if x is not None and torch.isnan(F.log_softmax(x, dim=1)).any():
            print('Nan in softmax')
            breakpoint()
        return F.log_softmax(x, dim=1)

    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad = torch.zeros(p.grad.shape, device=self.device)
        return

    def get_gradients(self):
        gradients = []
        for i, p in enumerate(self.parameters()):
            if p.grad is not None and torch.isnan(p.grad).any():
                print("NAN GRADIENT")
                breakpoint()

            if p.grad is not None and (torch.max(p.grad) > 1e4 or torch.min(p.grad) < -1e4):
                print("GRADIENT EXPLOSION")
                breakpoint()
            gradients.append(p.grad)
        # exit(0)
        return gradients

    def update(self, gradients, learning_rate=0.1):
        for i, p in enumerate(self.parameters()):
            if gradients[i] is not None:
                p.data -= learning_rate*gradients[i]

        return
