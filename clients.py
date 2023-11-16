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
        self.clients = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for i in range(self.clients_num):
            self.clients.append(Client(self, model().to(self.device)))



class Client:
    def __init__(self, group, model):
        """
        initializing a client
        group: the group that the client belongs to
        model: the model that the client uses
        """

        self.group = group
        self.model = model
        group_dataset = self.group.dataset
        self.dataset = Dataset(group_dataset.name, group_dataset.batch_size_train,
                               group_dataset.batch_size_test, group_dataset.flipped)
