import wandb
from clients import ClientGroup
from Models.smallnet import SmallNet
from train import Train
from datasets import Dataset
import argparse
from grouping import Grouping
from Models.resnet import ResNet9
from torch.utils.data import random_split

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs_train', type=int, default=128)
    parser.add_argument('--bs_test', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_lambda', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=50)
    parser.add_argument('--epochs',  type=int, default=40)
    parser.add_argument('--workers', type=int, nargs='+', default=[2, -1])
    parser.add_argument('--shared_layers', type=int, nargs='+', default=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    parser.add_argument('--wandb',type=bool, default=False)
    parser.add_argument('--train_method', type=str, default='shared_model_weighted_gradient_averaging')
    parser.add_argument('--eval_method', type=str, default='shared_model_evaluation')
    parser.add_argument('--grouping_method', type=str, default='frank_wolfe_update_grouping')
    parser.add_argument('--known_grouping', type=bool, default=False)
    parser.add_argument('--rho', type=float, default=0.01)
    parser.add_argument('--wandb_key', type=str, default='772bb501917cdea510fc4f46258987769788e1c3')
    parser.add_argument('--dataset', type=str, default='fashion_MNIST')
    parser.add_argument('--partitioned', type=bool, default=False)
    parser.add_argument('--identical_partition', type=bool, default=False)

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
    dataset_name = args.dataset
    partitioned = args.partitioned
    identical_partition = args.identical_partition

    shared_layers = [0 for i in range(62)]
    for i in range(62):
        shared_layers[i] = 1

    split_set = [0.01, 0.01, 0.01, 0.97]

    print('known grouping?', known_grouping)
    if wandb_run:
        if wandb_key is not None:
            wandb.login(key=wandb_key)

        wandb.init(
            project="personalization",
            name = "FMNIST"+ str(workers) + " unbalanced partition" + str(split_set) + " - lr =" + str(learning_rate) + " - alpha = " + str(alpha) + "- bs =" + str(batch_size_train) + " - rho = " + str(rho),
            # "FMNIST [2 -1] exact W - sum smooth grad - bs = 64 alpha reduce 500 steps - lr_param =" + str(learning_rate) + " - rho = " + str(rho) + "- alpha = " + str(alpha),

            # track hyperparameters and run metadata
            config={
                "learning_rate": learning_rate,
                "epochs": n_epochs,
                "batch_size": batch_size_train,
                "workers": workers,
                "shared_layers": shared_layers,
                "lr_lambda": lr_lambda,
                "alpha": alpha,
                "rho": rho,
                "split set": split_set
            }
        )

    worker_groups = list()
    used_net = SmallNet if dataset_name == 'fashion_MNIST' else ResNet9
    cnt_clients = 0

    print('used net: ', used_net)

    identical_dataset = Dataset(dataset_name, batch_size_train, batch_size_test)


    if identical_partition:
        train_data = identical_dataset.train_loader.dataset
        train_size = len(train_data)
        num_clients = abs(workers[0])
        print('train size', train_size)
        identical_partition = random_split(range(train_size), [int(train_size*x) for x in split_set])

    for i in range(len(workers)):
        flipped = workers[i] < 0
        cnt_clients += abs(workers[i])
        dataset = Dataset(dataset_name, batch_size_train, batch_size_test, flipped=flipped, seed=i*11)
        if partitioned:
            print('dataset is partitioned')
            if identical_partition:
                print('partitioning is identical', dataset)
                # dataset = Dataset.clone_trainset(identical_dataset)
                dataset = Dataset.partition_train(dataset, abs(workers[i]), indices=identical_partition)
                print('number of batches:', len(dataset.train_loader[0]))
                # if flipped:
                #     dataset.flipped = True
                #     dataset.seed = i*11
                #     Dataset.flip_labels(dataset.train_loader, dataset.test_loader, dataset.seed)
                # print('number of batches after flipping:', len(dataset.train_loader[0]))
            else:
                print('dataset is not partitioned')
                dataset = Dataset.partition_train(dataset, abs(workers[0]))

        print('here dataset type and trainloader type are:', type(dataset), type(dataset.train_loader), dataset, dataset.train_loader)
        worker_groups.append(ClientGroup(abs(workers[i]), used_net, batch_size_train, batch_size_test, dataset= dataset))

    grouping = Grouping(cnt_clients, lr_lambda, alpha, rho)
    # breakpoint()



    train = Train(worker_groups, learning_rate, known_grouping, shared_layers=shared_layers, grouping=grouping)
    train.train(getattr(train, train_method), getattr(train, eval_method), n_epochs, getattr(grouping, grouping_method))
