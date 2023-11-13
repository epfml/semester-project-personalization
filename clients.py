from datasets import Dataset
import torch
class ClientGroup:
    def __init__(self, clients_num, model, batch_size_train, batch_size_test, dataset):
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
        self.group = group
        self.model = model
        group_dataset = self.group.dataset
        self.dataset = Dataset(group_dataset.name, group_dataset.batch_size_train,
                               group_dataset.batch_size_test, group_dataset.flipped)
