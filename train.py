import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import math
class Train:
    """
    This is a class that implements different training methods
    groups: We know the grouping of all clients, which means that they have the same dataset to train on
    shared_layer: this is a mask of layers that are shared between all the clients. Other layers are only shared within each group
    """

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
        """finds the average of the model parameters for the models given in the input"""
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
        """finds the average of the model gradients for the models given in the input"""
        aggregated_grads = list()
        for i, model in enumerate(averaging_models):
            for j, grad in enumerate(model.get_gradients()):
                # if j == 0:
                #     print(grad[0,0,0,:3])
                if i == 0:
                    aggregated_grads.append(grad.clone())
                else:
                    aggregated_grads[j] += grad.clone()

        for i in range(len(aggregated_grads)):
            aggregated_grads[i] /= len(averaging_models)
            # if i == 0:
            #     print('averaged: ', aggregated_grads[i][0, 0, 0,:3])


        return aggregated_grads

    def gradient_averaging(self):
        """
        the simplest federated learning aggregation. Just average the gradients of all models (for all layers)
        and update the models based on that (we don't use shared layers here)
        """
        aggregated_grads = self.find_average_gradients(self.models)
        for model in self.models:
            model.update(aggregated_grads, self.learning_rate)


    def average_layer_parameters(self):
        """
        instead of aggregating gradient, each model, updates itself based on local gradient, and then we aggregate
        model parameters (we don't use shared layers here)
        """
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
        """
        We average the gradient of shared layers for all the models. For other layers we only average within their groups.
        The aggregated gradient of private layers is weighted by the number of clients in each group
        """
        average_grad_all = self.find_average_gradients(self.models)

        for model in self.models:
            for i, param in enumerate(model.parameters()):
                if self.shared_layers[i] == 1:
                    param.data -= self.learning_rate * average_grad_all[i]

        # return
        for group in self.groups:
            group_rate = len(group.clients)/len(self.models)
            models = list()
            for client in group.clients:
                models.append(client.model)

            average_grad_group = self.find_average_gradients(models)

            for j , client in enumerate(group.clients):
                for i, param in enumerate(client.model.parameters()):
                    if self.shared_layers[i] == 0:
                        param.data -= self.learning_rate * group_rate * average_grad_group[i]


    def first_client_evaluation(self):
        """evaluating accuracy and loss based on the first client model and dataset"""
        self.models[0].eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loaders[0]:
                output = self.models[0](data.to(self.device))
                test_loss += F.nll_loss(output, target.to(self.device), reduction='sum').item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.to(self.device).data.view_as(pred)).sum()
        test_loss /= len(self.test_loaders[0].dataset)
        test_acc = 100. * correct / len(self.test_loaders[0].dataset)

        wandb.log({"accuracy": test_acc, "loss": test_loss})
        print(test_acc)

    def shared_model_evaluation(self):
        """evaluating accuracy and loss based on average of accuracy and loss of all agents"""
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
                    test_loss += F.nll_loss(output, target.to(self.device), reduction='sum').item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.to(self.device).data.view_as(pred)).sum()

            test_loss /= len(self.test_loaders[i].dataset)
            test_acc = 100. * correct / len(self.test_loaders[i].dataset)
            print('client ', i, 'acc: ', test_acc, 'loss: ', test_loss)
            global_loss += test_loss
            global_acc += test_acc
        if math.isnan(global_loss / len(self.models)):
            breakpoint()
        print('global acc:', global_acc/len(self.models), 'global loss: ', global_loss/len(self.models))
        wandb.log({"accuracy": global_acc/len(self.models), "loss": global_loss/len(self.models)})

    def train(self, aggregator, evaluate, epochs):
        """
        main training loop
        aggregator: one of the methods for aggregating gradients/parameters and updating the models
        evaluate: one of the methods for evaluating the models
        """

        if aggregator == self.shared_model_weighted_gradient_averaging:
            self.average_layer_parameters()

        cnt = 0
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

                cnt += 1
                if cnt % 250 == 0:
                    evaluate()
