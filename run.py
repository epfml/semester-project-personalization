import wandb
import torch
from clients import ClientGroup
from model import Net
from train import Train
from datasets import Dataset
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.1
n_epochs = 10


wandb.init(
    project="personalization",
    name = "2 FMNIST + 1 FMNIST-F - shared model - 1CL",
    # track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "epochs": n_epochs,
        "batch_size": batch_size_train
    }
)

g1 = ClientGroup(2, Net, batch_size_train, batch_size_test, dataset= Dataset('load_Fashion_MNIST', batch_size_train, batch_size_test))
g2 = ClientGroup(1, Net, batch_size_train, batch_size_test, dataset= Dataset('load_Fashion_MNIST', batch_size_train, batch_size_test), flipped=True)

# print('flipped dataset:', g2.clients[0].dataset.train_loader.dataset.targets)
print('normal dataset:', g1.clients[0].dataset.train_loader.dataset.targets)
train = Train([g1, g2], learning_rate, shared_layers=[1, 1, 0, 0, 0, 0, 0, 0])
train.train(train.shared_model_weighted_gradient_averaging, train.shared_model_evaluation, n_epochs)
