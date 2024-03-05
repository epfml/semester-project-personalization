import torch
import torch.nn.functional as F
import wandb
import os

class Train:
    """
    This is a class that implements different training methods
    groups: We know the grouping of all clients, which means that they have the same dataset to train on
    shared_layer: this is a mask of layers that are shared between all the clients. Other layers are only shared within each group
    """

    def __init__(self, groups, learning_rate, known_grouping, master_process, shared_layers = None, grouping = None):
        self.test_grad = None
        self.groups = groups
        self.learning_rate = learning_rate
        self.known_grouping = known_grouping
        self.train_loaders, self.val_loaders, self.test_loaders = list(), list(), list()
        self.models, self.clients = list(), list()
        self.initial_models = list()
        self.grouping = grouping
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_losses = torch.zeros(len(self.models), device=self.device)
        self.master_process = master_process

        for group in self.groups:
            for client in group.clients:
                self.clients.append(client)
                self.models.append(client.model)

                initial_model = client.group.model()
                initial_model.load_state_dict(client.model.state_dict())
                self.initial_models.append(initial_model.to(self.device))
                self.train_loaders.append(client.dataset.train_loader)
                self.test_loaders.append(client.dataset.test_loader)
                self.val_loaders.append(client.dataset.val_loader)



        if not shared_layers is None:
            self.shared_layers = shared_layers


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


    def find_average_momentum(self, averaging_models):
        """finds the average of the model gradients for the models given in the input"""
        aggregated_momentum = list()
        for i, model in enumerate(averaging_models):
            for j, momentum in enumerate(model.get_momentum()):
                # if j == 0:
                #     pisrint(grad[0,0,0,:3])
                if i == 0:
                    aggregated_momentum.append(momentum.clone())
                else:
                    aggregated_momentum[j] += momentum.clone()

        for i in range(len(aggregated_momentum)):
            aggregated_momentum[i] /= len(averaging_models)
            # if i == 0:
            #     print('averaged: ', aggregated_grads[i][0, 0, 0,:3])


        return aggregated_momentum

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
            cnt = 0
            for j, param in enumerate(model.parameters()):
                # if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear) or isinstance(param, nn.BatchNorm2d):
                #     cnt += 1
                #     print('haha', cnt, param.data.shape)
                # #     param.data = average_params_all[j]
                # # print(j, param.data.shape)
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
        average_momentum_all = self.find_average_gradients(self.models)

        for model in self.models:
            for i, param in enumerate(model.parameters()):
                if self.shared_layers[i] == 1:
                    param.data -= self.learning_rate * average_momentum_all[i]

        # return
        for group in self.groups:
            group_rate = len(group.clients)/len(self.models)
            # group_rate = 1
            models = list()
            for client in group.clients:
                models.append(client.model)

            average_momentum_group = self.find_average_gradients(models)

            for j , client in enumerate(group.clients):
                for i, param in enumerate(client.model.parameters()):
                    if self.shared_layers[i] == 0:
                        param.data -= self.learning_rate * group_rate * average_momentum_group[i]


        for model in self.models:
            model.previous_momentum = model.get_momentum()


    def one_client_evaluation(self, model, test_loader):
        """evaluating accuracy and loss based on the first client model and dataset"""
        model.zero_grad()
        model.eval()
        test_loss = 0
        correct = 0
        flag = False

        with torch.no_grad():
            for data, target in test_loader:
                output = model(data.to(self.device))
                test_loss += F.cross_entropy(output, target.to(self.device)).detach()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.to(self.device).data.view_as(pred)).sum()


        test_loss /= len(test_loader)
        test_acc = 100. * correct / len(test_loader.dataset)
        print('size of test set:', len(test_loader.dataset))
        # wandb.log({"accuracy": test_acc, "loss": test_loss})
        # print('accuracy and loss', test_acc, test_loss)
        return test_acc, test_loss


    def neighbors_gradient_averaging(self):
        average_grad_all = self.find_average_gradients(self.models)

        for model in self.models:
            for i, param in enumerate(model.parameters()):
                if self.shared_layers[i] == 1:
                    param.data -= self.learning_rate * average_grad_all[i]

        for client in self.clients:
            average_grad_neighbors = self.find_average_gradients(client.neighbor_models)
            group_rate = len(client.neighbor_models) / len(self.models)
            for i, param in enumerate(client.model.parameters()):
                if self.shared_layers[i] == 0:
                    param.data -= self.learning_rate * group_rate * average_grad_neighbors[i]

    def frank_wolfe_gradient_update(self):
        num_clients = len(self.clients)
        cloned_models = list()

        for client in self.clients:
            cloned = client.group.model()
            cloned.load_state_dict(client.model.state_dict())
            cloned_models.append(cloned.to(self.device))

        for ind1, client1 in enumerate(self.clients):
            for ind2, client2 in enumerate(self.clients):
                if client1 != client2:
                    less_ind = min(ind1, ind2)
                    more_ind = max(ind1, ind2)
                    models_params = (client1.model.parameters(), client2.model.parameters(),
                                     cloned_models[ind1].parameters(), cloned_models[ind2].parameters())
                    # unflatten_lambda = self.grouping.unflat_lambda(None if self.grouping.w_lambda is None else self.grouping.w_lambda[less_ind, more_ind], client1, self)
                    for param_ind, (p1, p2, p1_clone, p2_clone) in enumerate(zip(*models_params)):
                        if self.shared_layers[param_ind] == 1:

                            p1.data -= (self.learning_rate * self.grouping.rho
                                        * self.grouping.w_adjacency[less_ind, more_ind] * (p1_clone.data - p2_clone.data))
                            # sign = 1 if ind1 < ind2 else -1
                            # p1.data -= self.learning_rate * self.grouping.w_adjacency[less_ind, more_ind] * sign * unflatten_lambda[param_ind]


        # for i, client in enumerate(self.clients):
        #     for param_ind, (p1, cloned_p1, initial_p1) in enumerate(zip(client.model.parameters(), cloned_models[i].parameters(), self.initial_models[i].parameters())):
        #         if self.shared_layers[param_ind] == 1:
        #             p1.data -= 1 / num_clients * self.learning_rate * (cloned_p1.data - initial_p1.data)

        for client in self.clients:
            model = client.model
            momentum = model.get_momentum()
            for i, p in enumerate(model.parameters()):
                p.data -= 1/num_clients * self.learning_rate * momentum[i]

        for model in self.models:
            model.previous_momentum = model.get_momentum()

    def shared_model_evaluation(self, step):
        """evaluating accuracy and loss based on average of accuracy and loss of all agents"""
        global_loss = 0
        global_acc = 0
        for i, client in enumerate(self.clients):
            test_acc, test_loss = self.one_client_evaluation(client.model, client.dataset.test_loader)
            print('client ', i, "test accuracy and loss: ", test_acc, test_loss)
            global_loss += test_loss
            global_acc += test_acc

        print('global acc:', global_acc / len(self.models), 'global loss: ', global_loss / len(self.models))
        wandb.log({"accuracy": global_acc / len(self.models), "loss": global_loss / len(self.models)})

        return global_acc, global_loss

    def train(self, aggregator, evaluate, epochs, grouping_method=None,
              starting_epoch=0, partitioning=None, run_id=None):
        """
        main training loop
        aggregator: one of the methods for aggregating gradients/parameters and updating the models
        evaluate: one of the methods for evaluating the models
        """

        if aggregator == self.shared_model_weighted_gradient_averaging:
            self.average_layer_parameters()

        cnt = 0

        current_epoch = 0
        epochs_flag = [False for i in range(epochs*10)]

        while starting_epoch + current_epoch < epochs:
            min_client_epoch = float('inf')
            for client in self.clients:
                next_batch, client_epoch = next(client.next_batch)
                min_client_epoch = min(min_client_epoch, client_epoch)

                client.model.train()
                client.model.zero_grad()
                output = client.model(next_batch[0].to(self.device))
                loss = F.cross_entropy(output, next_batch[1].to(self.device))
                loss.backward()

            current_epoch = min_client_epoch

            groups_models = []
            groups_momentums = []

            for i in range(len(self.groups)):
                group_models = []
                group_momentums = []
                for j in range(len(self.groups[i].clients)):
                    group_models.append(self.groups[i].clients[j].model.state_dict())
                    group_momentums.append(self.groups[i].clients[j].model.previous_momentum)
                groups_models.append(group_models)
                groups_momentums.append(group_momentums)

            if self.master_process and not epochs_flag[current_epoch]:
                checkpoint_dict = {"current_epoch": current_epoch,
                                "models": groups_models,
                                "w_adjacency": self.grouping.w_adjacency,
                                "partitioning": partitioning,
                                "learning_rate": self.learning_rate,
                                "momentum": groups_momentums,
                                "starting_epoch": current_epoch
                }

                file_path = "/mloscratch/homes/hashemi/semester-project-personalization/checkpoints/"
                print("file path:", file_path)
                torch.save(checkpoint_dict, file_path + run_id + '_last.pt')
                wandb.save(file_path + run_id + '_last.pt')

                if current_epoch % 5 == 0:
                    self.learning_rate *= 0.9
                    torch.save(checkpoint_dict, file_path + run_id + '_epoch' + str(current_epoch) + '.pt')
                    wandb.save(file_path + run_id + '/epoch' + str(current_epoch) + '.pt')

            epochs_flag[current_epoch] = True

            if not self.known_grouping:
                grouping_method(self.clients, self)

            aggregator()
            cnt += 1

            if cnt % 500 == 0 and self.master_process:
                evaluate(cnt//500)
