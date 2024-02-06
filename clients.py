from datasets import Dataset
import torch
class ClientGroup:
    def __init__(self, clients_num, model, batch_size_train, batch_size_test, dataset):
        """
        builds a group of clients with the same dataset
        clients_num: number of clients in the group
        model: the model that the clients will use
        dataset: the dataset that the clients will use
        """

        self.clients_num = clients_num
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.dataset = dataset
        print('dataset is:', self.dataset)
        print('train loader type is:', type(self.dataset.train_loader))
        print('number of batches in client group:', len(self.dataset.train_loader[0]))
        self.clients = []
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        counter1 = []
        counter2 = []
        for i in range(10):
            counter1.append(0)
            counter2.append(0)


        for _,labels in self.dataset.train_loader[0]:
            for label in labels:
                counter1[label] += 1

        for _, labels in self.dataset.train_loader[1]:
            for label in labels:
                counter2[label] += 1

        print('counters are:', counter1, counter2)


        for i in range(self.clients_num):
            self.clients.append(Client(self, model, i))



class Client:
    def __init__(self, group, model, index):
        """
        initializing a client
        group: the group that the client belongs to
        model: the model that the client uses
        """
        self.neighbor_models = list()
        self.neighbor_inds = list()
        self.group = group
        self.model = model(self).to(self.group.device)
        group_dataset = self.group.dataset
        print(group_dataset.flipped)
        self.dataset = Dataset(group_dataset.name, group_dataset.batch_size_train,
                               group_dataset.batch_size_test, group_dataset.flipped, group_dataset.seed)

        if isinstance(self.group.dataset.train_loader, list):
            self.dataset.train_loader = self.group.dataset.train_loader[index]

        self.next_batch = self.get_next_batch(self.dataset.train_loader)


    def get_next_batch(self, train_loader):
        num_epoch = 0
        while True:
            for data in train_loader:
                yield data, num_epoch
            num_epoch += 1