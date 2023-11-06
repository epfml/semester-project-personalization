import torch
import torch.nn as nn
import torch.nn.functional as F
from statistics import mean
from tqdm import tqdm
import wandb
import numpy as np
class Train:
    def __init__(self, groups, learning_rate, shared_layers = None):
        self.test_grad = None
        self.groups = groups
        self.learning_rate = learning_rate
        self.train_loaders, self.test_loaders, self.models = list(), list(), list()
        for group in self.groups:
            for client in group.clients:
                self.models.append(client.model)
                self.train_loaders.append(client.dataset.train_loader)
                self.test_loaders.append(client.dataset.test_loader)

        if not shared_layers is None:
            self.shared_layers = shared_layers

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def find_average_parameters(self, averaging_models):
        params = list()
        for i in range(len(averaging_models)):
            for j, param in enumerate(averaging_models[i].parameters()):
                if i == 0:
                    params.append(param.data.clone())
                else:
                    params[j] += param.data.clone()

        for i in range(len(params)):
            params[i] /= len(averaging_models)

        return params

    def find_average_gradients(self, averaging_models):
        aggregated_grads = list()
        for i, model in enumerate(averaging_models):
            for j, grad in enumerate(model.get_gradients()):
                if i == 0:
                    aggregated_grads.append(grad.clone())
                else:
                    aggregated_grads[j] += grad.clone()

        for i in range(len(aggregated_grads)):
            aggregated_grads[i] /= len(averaging_models)

        return aggregated_grads

    def gradient_averaging(self):
        aggregated_grads = self.find_average_gradients(self.models)
        for model in self.models:
            model.update(aggregated_grads, self.learning_rate)


    def average_layer_parameters(self):

        for model in self.models:
            model.update(model.get_gradients(), self.learning_rate)

        average_params_all = self.find_average_parameters(self.models)

        for model in self.models:
            for j, param in enumerate(model.parameters()):
                if self.shared_layers[j] == 1:
                    param.data = average_params_all[j]

        for group in self.groups:
            group_models = list()
            for client in group.clients:
                group_models.append(client.model)

            average_params_group = self.find_average_parameters(group_models)

            for model in group_models:
                for j, param in enumerate(model.parameters()):
                    if self.shared_layers[j] == 0:
                        param.data = average_params_group[j]


    def shared_model_weighted_gradient_averaging(self):
        average_grad_all = self.find_average_gradients(self.models)

        for model in self.models:
            for i, param in enumerate(model.parameters()):
                if self.shared_layers[i] == 1:
                    param.data -= self.learning_rate*average_grad_all[i]

        # return
        for group in self.groups:
            group_rate = len(group.clients)/len(self.models)
            # print(group_rate)
            models = list()
            for client in group.clients:
                models.append(client.model)

            average_grad_group = self.find_average_gradients(models)

            for j , client in enumerate(group.clients):
                for i, param in enumerate(client.model.parameters()):
                    if self.shared_layers[i] == 0:
                        param.data -= self.learning_rate * group_rate * average_grad_group[i]


        # breakpoint()
    def first_client_evaluation(self):
        self.models[0].eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loaders[0]:
                output = self.models[0](data.to(self.device))
                test_loss += F.nll_loss(output, target.to(self.device), size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.to(self.device).data.view_as(pred)).sum()
        test_loss /= len(self.test_loaders[0].dataset)
        test_acc = 100. * correct / len(self.test_loaders[0].dataset)

        wandb.log({"accuracy": test_acc, "loss": test_loss})
        print(test_acc)

    def shared_model_evaluation(self):
        global_loss = 0
        global_acc = 0
        for i, model in enumerate(self.models):
            model.zero_grad()
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in self.test_loaders[i]:
                    output = model(data.to(self.device))
                    test_loss += F.nll_loss(output, target.to(self.device), size_average=False).item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.to(self.device).data.view_as(pred)).sum()

            test_loss /= len(self.test_loaders[i].dataset)
            test_acc = 100. * correct / len(self.test_loaders[i].dataset)
            print('client ', i, 'acc: ', test_acc, 'loss: ', test_loss)
            global_loss += test_loss
            global_acc += test_acc

        print('global acc:', global_acc/len(self.models), 'global loss: ', global_loss/len(self.models))
        wandb.log({"accuracy": global_acc/len(self.models), "loss": global_loss/len(self.models)})

    def train(self, aggregator, evaluate, epochs):
        for epoch in tqdm(range(epochs)):
            for batches in zip(*self.train_loaders):
                model_ind = 0
                for (data, target) in batches:
                    model = self.models[model_ind]
                    model.train()
                    model.zero_grad()
                    output = model(data.to(self.device))
                    loss = model.criterion(output, target.to(self.device))
                    loss.backward()
                    model_ind += 1

                aggregator()

            evaluate()

# def evaluate(model, test_dataset):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_dataset:
#             output = model(data.to(device))
#             test_loss += F.nll_loss(output, target.to(device), size_average=False).item()
#             pred = output.data.max(1, keepdim=True)[1]
#             correct += pred.eq(target.to(device).data.view_as(pred)).sum()
#     test_loss /= len(test_dataset.dataset)
#     test_acc = 100. * correct / len(test_dataset.dataset)
#     wandb.log({"accuracy": test_acc, "loss": test_loss})
#     print(test_acc)
#
# def centralized_train(model, test_dataset, epochs, aggregator, *datasets):
#     for epoch in tqdm(range(epochs)):
#         model.train()
#         for batches in zip(*datasets):
#             grads = list()
#             # print(len(batches), len(batches[0]))
#             for (data, target) in batches:
#                 model.zero_grad()
#                 output = model(data.to(device))
#                 loss = model.criterion(output, target.to(device))
#                 loss.backward()
#                 grads.append(model.get_gradients())
#             aggregated_grads = aggregator(grads)
#             model.update(aggregated_grads)
#
#         evaluate(model, test_dataset)
#     return
#
#
# def decentralized_train(test_model, models, test_dataset, epochs, aggregator, datasets):
#     for epoch in tqdm(range(epochs)):
#         for i, model in enumerate(models):
#             model.train()
#             for (data, target) in datasets[i]:
#                 model.zero_grad()
#                 output = model(data.to(device))
#                 loss = model.criterion(output, target.to(device))
#                 loss.backward()
#                 model.update(model.get_gradients())
#
#         average_models(models)
#         evaluate(test_model, test_dataset)
#
#
# def average_models(models):
#     params = list()
#     for i in range(len(models)):
#         for j, param in enumerate(models[i].parameters()):
#             if i == 0:
#                 params.append(param.data)
#             elif i < 2 or 4 <= j <= 5:
#                 params[j] += param.data
#
#     # print(len(params))
#     for i in range(len(params)):
#         # print(len(models))
#         # params[i] /= len(models)
#         print(params[i].shape)
#         # if 4 <= i <= 5:
#         #     params[i] /= 3
#         # else:
#         #     params[i] /= 2
#
#     for i in range(len(models)):
#         model = models[i]
#         for j, param in enumerate(model.parameters()):
#             # param.data = params[j]
#             if 4 <= j <= 5 or i+1 < len(models):
#                 param.data = params[j]/len(models)
#             else:
#                 param.data /= len(models)
#
#     return
#     # print(params)
#     # exit(0)
#
#
# def averaging(gradients):
#     grads = []
#     for i in range(len(gradients[0])):
#         grads.append(gradients[0][i])
#         for j in range(1, len(gradients)):
#             grads[i] += gradients[j][i]
#
#         grads[i]/= len(gradients)
#
#     return grads



# batch_size_train = 64
# batch_size_test = 1000

#
# train_loader1, test_loader1 = load_Fashion_MNIST(batch_size_train, batch_size_test)
# train_loader2, test_loader2 = load_Fashion_MNIST(batch_size_train, batch_size_test)
# train_loader3, test_loader3 = load_Fashion_MNIST(batch_size_train, batch_size_test)
#
# # print(train_loader3.dataset.targets)
# train_loader2.dataset.targets = (train_loader2.dataset.targets + 5) % 10
# # print(train_loader3.dataset.targets)
#
#
# model1 = model.Net().to(device)
# model2 = modelNet().to(device)
# model3 = Net().to(device)
# # print(device, type(train_loader1))
#
# decentralized_train(model1, [model1, model2, model3], test_loader1, n_epochs, averaging, [train_loader1, train_loader2, train_loader3])