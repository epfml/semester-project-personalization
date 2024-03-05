import wandb
from clients import ClientGroup
from Models.smallnet import SmallNet
from train import Train
from datasets import Dataset
import argparse
from grouping import Grouping
from Models.resnet import ResNet9
from torch.utils.data import random_split
import torch
import os
from torch.distributed import init_process_group

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--bs_train', type=int, default=128)
    parser.add_argument('--bs_test', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_lambda', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=50)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--workers', type=int, nargs='+', default=[2, -1])
    parser.add_argument('--shared_layers', type=int, nargs='+',
                        default=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--train_method', type=str, default='shared_model_weighted_gradient_averaging')
    parser.add_argument('--eval_method', type=str, default='shared_model_evaluation')
    parser.add_argument('--grouping_method', type=str, default='frank_wolfe_update_grouping')
    parser.add_argument('--known_grouping', type=bool, default=False)
    parser.add_argument('--rho', type=float, default=0.01)
    parser.add_argument('--wandb_key', type=str, default='772bb501917cdea510fc4f46258987769788e1c3')
    parser.add_argument('--dataset', type=str, default='fashion_MNIST')
    parser.add_argument('--partitioned', type=bool, default=False)
    parser.add_argument('--identical_partition', type=bool, default=False)
    parser.add_argument('--wandb_id', type=str, default=wandb.util.generate_id())
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
    wandb_id = args.wandb_id

    shared_layers = [0 for i in range(62)]
    for i in range(62):
        shared_layers[i] = 1

    split_set = [0.25, 0.25, 0.25, 0.25]

    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend='nccl')  # Should I change it?
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        device = f'cuda:{0}'
        print('ddp_rank', ddp_rank, '\nddp_local_rank', ddp_local_rank)
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0

    run_name = "Cifar100" + str(workers) + "_alpha_drop_partition_" + str(split_set) + "_lr:" + str(
        learning_rate) + "_alpha:" + str(alpha) + "_bs:" + str(batch_size_train) + "_rho:" + str(rho)
    print('wandb ID is:', wandb_id)

    if wandb_run and master_process:
        if wandb_key is not None:
            wandb.login(key=wandb_key)

        wandb.init(
            project="personalization",
            name="new run name!",
            id=wandb_id,
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
    dir_checkpoint = "/mloscratch/homes/hashemi/semester-project-personalization/checkpoints/" + run_name + "_last.pt"

    try:
        checkpoint = torch.load(dir_checkpoint)
    except FileNotFoundError:
        print('checkpoint file does not found')

    identical_dataset = Dataset(dataset_name, batch_size_train, batch_size_test)
    partitioning = None
    if identical_partition:
        if wandb.run is not None and wandb.run.resumed:
            partitioning = checkpoint["partitioning"]
        else:
            train_size = len(identical_dataset.train_loader.dataset)
            partitioning = random_split(range(train_size), [int(train_size * x) for x in split_set])

    for i in range(len(workers)):
        flipped = workers[i] < 0
        cnt_clients += abs(workers[i])
        dataset = Dataset(dataset_name, batch_size_train, batch_size_test, flipped=flipped, seed=i * 11)
        if partitioned:
            print('dataset is partitioned')
            if identical_partition:
                print('partitioning is identical', dataset)
                dataset = Dataset.partition_train(dataset, abs(workers[i]), indices=partitioning)
                print('number of batches:', len(dataset.train_loader[0]))
            else:
                print('dataset is not partitioned')
                dataset = Dataset.partition_train(dataset, abs(workers[0]))

        worker_groups.append(ClientGroup(abs(workers[i]), used_net, batch_size_train, batch_size_test, dataset=dataset))

    grouping = Grouping(cnt_clients, alpha, rho)
    starting_epoch = 0

    if wandb.run is not None and wandb.run.resumed:
        print('Resuming the training')
        # wandb.restore("last.pt")
        for i in range(len(workers)):
            for j in range(abs(workers[i])):
                worker_groups[i].clients[j].model.load_state_dict(checkpoint['models'][i][j])

                worker_groups[i].clients[j].model.previous_momentum = checkpoint['momentum'][i][j]

        # alpha = checkpoint['alpha']
        grouping = Grouping(cnt_clients, alpha, rho, w_adjacency=checkpoint['w_adjacency'])
        # starting_epoch = checkpoint["starting_epoch"]
        learning_rate = checkpoint["learning_rate"]

    train = Train(worker_groups, learning_rate, known_grouping, master_process, shared_layers=shared_layers, grouping=grouping)
    train.train(getattr(train, train_method), getattr(train, eval_method), n_epochs, getattr(grouping, grouping_method),
                starting_epoch=starting_epoch, partitioning=partitioning, run_id=wandb_id)
