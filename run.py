import wandb
from clients import ClientGroup
from model import Net
from train import Train
from datasets import Dataset
import argparse
from grouping import Grouping

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs_train', type=int, default=64)
    parser.add_argument('--bs_test', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--lr_lambda', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--epochs',  type=int, default=20)
    parser.add_argument('--workers', type=int, nargs='+', default=[2, -1])
    parser.add_argument('--shared_layers', type=int, nargs='+', default=[1, 1, 1, 1, 1, 1, 1, 1])
    parser.add_argument('--wandb',type=bool, default=False)
    parser.add_argument('--train_method', type=str, default='frank_wolfe_gradient_update')
    parser.add_argument('--eval_method', type=str, default='shared_model_evaluation')
    parser.add_argument('--grouping_method', type=str, default='frank_wolfe_update_grouping')
    parser.add_argument('--known_grouping', type=bool, default=False)
    parser.add_argument('--rho', type=float, default=0.005)
    parser.add_argument('--wandb_key', type=str, default=None)


    args = parser.parse_args()
    batch_size_train = args.bs_train
    batch_size_test = args.bs_test
    learning_rate = args.lr
    lr_lambda = args.lr_lambda
    n_epochs = args.epochs
    workers = args.workers
    shared_layers = args.shared_layers
    wandb_run = args.wandb
    train_method = args.train_method
    eval_method = args.eval_method
    known_grouping = args.known_grouping
    grouping_method = args.grouping_method
    alpha = args.alpha
    rho = args.rho
    wandb_key = args.wandb_key

    print('known grouping?', known_grouping)
    if wandb_run:
        if wandb_key is not None:
            wandb.login(key=wandb_key)

        wandb.init(
            project="personalization",
            name = "FMNIST [2 -1] exact W - sum smooth grad - bs = 64 alpha reduce 500 steps - lr_param =" + str(learning_rate) + " - rho = " + str(rho) + "- alpha = " + str(alpha),
            # track hyperparameters and run metadata
            config={
                "learning_rate": learning_rate,
                "epochs": n_epochs,
                "batch_size": batch_size_train,
                "workers": workers,
                "shared_layers": shared_layers,
                "lr_lambda": lr_lambda,
                "alpha": alpha,
                "rho": rho
            }
        )

    worker_groups = list()

    cnt_clients = 0
    for i in range(len(workers)):
        flipped = workers[i] < 0
        cnt_clients += abs(workers[i])
        worker_groups.append(ClientGroup(abs(workers[i]), Net, batch_size_train, batch_size_test,
                            dataset= Dataset('fashion_MNIST', batch_size_train, batch_size_test, flipped=flipped, seed = i)))
    grouping = Grouping(cnt_clients, lr_lambda, alpha, rho)
    # breakpoint()



    train = Train(worker_groups, learning_rate, known_grouping, shared_layers=shared_layers, grouping=grouping)
    train.train(getattr(train, train_method), getattr(train, eval_method), n_epochs, getattr(grouping, grouping_method))
