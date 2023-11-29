from model import Net
import math
import torch
class Grouping:

    def __init__(self, n_clients, lr_lambda=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_clients = n_clients
        self.w_adjacency = torch.eye(self.n_clients, device=self.device)
        self.w_lambda = torch.eye(self.n_clients, device=self.device)
        self.lr_lambda = lr_lambda
        self.step = 0

    def frank_wolfe_update_grouping(self, clients, train):
        """Calculate the matrix norm_differences. The (i,j)-th entry is the l2 norm of difference between i-th and j-th model shared layers"""
        # cnt = 0
        flattened_params_list = list()
        for client in clients:
            flattened_model = list()
            for i, param in enumerate(client.model.parameters()):
                # if cnt < 2 and i == 0:
                #     print('first layer param is: ', param)
                #     cnt += 1
                if train.shared_layers[i] == 1:
                    flattened_model.append(torch.reshape(param.clone(), (-1,)))

            flattened_params_list.append(torch.cat(flattened_model))

        stacked_params = torch.stack(flattened_params_list)
        # print('stacked params for first and second clients:', stacked_params[0, :10], stacked_params[1, :10])
        norm_squared_differences = torch.norm(stacked_params[:, None, :] - stacked_params[None, :, :], p=2, dim = -1)
        # print('real norm of first 2', torch.norm(stacked_params[0,:] - stacked_params[1,:], p=2))
        # print('computed norm of first 2', norm_squared_differences[0,1])

        """update w_lambda"""
        self.w_lambda += self.lr_lambda * 2* self.w_adjacency * norm_squared_differences

        """update w_adjacency"""
        step_size_frank_wolfe = 2 / (self.step + 2)
        s = torch.zeros(self.w_adjacency.shape, device=self.device)
        s[self.w_lambda * norm_squared_differences < 0] = 1
        self.w_adjacency += step_size_frank_wolfe * (s - self.w_adjacency)

        self.step += 1



    def average_loss_grouping(self, clients, train):
        for client in clients:
            client.neighbor_models = list()
            client.neighbor_inds = list()

        accs, losses = list(), list()

        for i, client in enumerate(clients):
            acc, loss = train.one_client_evaluation(client.model, client.dataset.val_loader)
            accs.append(acc)
            losses.append(loss)

        for i, client1 in enumerate(clients):
            client1.neighbor_models.append(client1.model)
            for j, client2 in enumerate(clients):
                if i < j:
                    avg_model = Net()
                    avg_model.set_params(train.find_average_parameters([client1.model, client2.model]))
                    # print('avg eval1: ')
                    avg_acc1, avg_loss1 = train.one_client_evaluation(avg_model, client1.dataset.val_loader)
                    # print('avg eval2:')
                    avg_acc2, avg_loss2 = train.one_client_evaluation(avg_model, client2.dataset.val_loader)
                    # print('client1 eval: ')
                    # acc1, loss1 = train.one_client_evaluation(client1.model, client1.dataset.val_loader)
                    # print('client2 eval: ')
                    # acc2, loss2 = train.one_client_evaluation(client2.model, client2.dataset.val_loader)
                    acc1, loss1 = accs[i], losses[i]
                    acc2, loss2 = accs[j], losses[j]
                    print('avg accuracy and loss', (avg_acc1 + avg_acc2)/2, (avg_loss1 + avg_loss2)/2, avg_loss1, avg_loss2)
                    print('client 1 accuracy and loss', acc1, loss1)
                    print('client 2 accuracy and loss', acc2, loss2)

                    # if avg_loss1 + avg_loss2 > loss1 + loss2 and avg_acc1 + avg_acc2 > acc1 + acc2:
                    #     breakpoint()

                    if (avg_acc1 + avg_acc2)/2 > (acc1 + acc2)/2:
                        client1.neighbor_models.append(client2.model)
                        client2.neighbor_models.append(client1.model)
                        client1.neighbor_inds.append(j)
                        client2.neighbor_inds.append(i)

