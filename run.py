import wandb
import torch
from clients import ClientGroup
from model import Net
from train import Train
from datasets import Dataset
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs_train', type=int, default=16)
    parser.add_argument('--bs_test', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--epochs',  type=int, default=10)
    parser.add_argument('--workers', type=int, nargs='+', default=[2, -1])
    parser.add_argument('--shared_layers', type=int, nargs='+', default=[1, 1, 1, 1, 0, 0, 0, 0])
    parser.add_argument('--wandb',type=bool, default=False)
    parser.add_argument('--train_method', type=str, default='shared_model_weighted_gradient_averaging')
    parser.add_argument('--eval_method', type=str, default='shared_model_evaluation')

    args = parser.parse_args()
    batch_size_train = args.bs_train
    batch_size_test = args.bs_test
    learning_rate = args.lr
    n_epochs = args.epochs
    workers = args.workers
    shared_layers = args.shared_layers
    wandb_run = args.wandb
    train_method = args.train_method
    eval_method = args.eval_method

    if wandb_run:
        wandb.init(
            project="personalization",
            name = "FMNIST[2 -1], bs=16, shared, all CL (w/o similar params)",
            # track hyperparameters and run metadata
            config={
                "learning_rate": learning_rate,
                "epochs": n_epochs,
                "batch_size": batch_size_train,
                "workers": workers,
                "shared_layers": shared_layers
            }
        )

    worker_groups = list()

    for i in range(len(workers)):
        flipped = workers[i] < 0
        worker_groups.append(ClientGroup(abs(workers[i]), Net, batch_size_train, batch_size_test,
                            dataset= Dataset('fashion_MNIST', batch_size_train, batch_size_test, flipped=flipped)))


    train = Train(worker_groups, learning_rate, shared_layers=shared_layers)
    train.train(getattr(train, train_method), getattr(train, eval_method), n_epochs)
