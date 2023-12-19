from model import Net
import math
import torch
import wandb
class Grouping:

    def __init__(self, n_clients, lr_lambda=None, alpha=None, rho=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_clients = n_clients
        self.w_adjacency = torch.ones((self.n_clients, self.n_clients), device=self.device)
        self.w_lambda = None
        self.lr_lambda = lr_lambda
        self.step = 0
        self.alpha = alpha
        self.rho = rho
        self.sum_grad = None
        self.patience = 0
        self.differences = None
        self.prv = 0
        self.norm_squared_differences = None
        self.alpha_encourage = 0
        print('lr lambda:', self.lr_lambda, 'alpha:', self.alpha)

    def frank_wolfe_update_grouping(self, clients, train):
        """Calculate the matrix norm_differences. The (i,j)-th entry is the l2 norm of difference between i-th and j-th model shared layers"""
        flattened_params_list = list()
        flattened_grads_list = list()
        for client in clients:
            flattened_model = list()
            flattened_grad = list()
            for i, param in enumerate(client.model.parameters()):
                if train.shared_layers[i] == 1:
                    flattened_model.append(torch.reshape(param.data.clone(), (-1,)))
                    flattened_grad.append(torch.reshape(param.grad.clone(), (-1,)))


            flattened_params_list.append(torch.cat(flattened_model))
            flattened_grads_list.append(torch.cat(flattened_grad))


        """alpha scheduler - we ignore this part for now"""
        stacked_gradients = torch.stack(flattened_grads_list)
        grad_norms = torch.norm(stacked_gradients, p=2, dim=-1)

        smooth_grad_norms = grad_norms

        if self.step != 0:
            smooth_grad_norms = 0.99*self.prv + 0.01*grad_norms

        self.prv = smooth_grad_norms
        sum_smooth_grad = torch.sum(smooth_grad_norms)
        # if self.step % 1000 == 0:
        #     self.alpha *= 0.9
        if self.step > 1000:
            if self.sum_grad is None or self.sum_grad > sum_smooth_grad:
                self.sum_grad = sum_smooth_grad
                self.patience = 0
                self.alpha_encourage += 1
                if self.alpha_encourage == 50:
                    self.alpha *= 1.1
                    self.alpha_encourage = 0
                    print('alpha increased! current alpha:', self.alpha)
            else:
                self.patience += 1

            if self.patience == 500:
                self.alpha_encourage = 0
                self.alpha *= 0.9
                self.patience = 0
                print('alpha redueced! current alpha:', self.alpha)

        stacked_params = torch.stack(flattened_params_list)
        self.differences = stacked_params[:, None, :] - stacked_params[None, :, :]

        self.norm_squared_differences = torch.norm(self.differences, p=2, dim = -1)**2

        """update w_lambda"""

        # if self.w_lambda is None:
        #     self.w_lambda = torch.zeros((self.n_clients, self.n_clients, stacked_params.shape[1]), device=self.device)
        #
        # self.w_lambda += self.lr_lambda * (self.w_adjacency[:, :, None] * differences)

        """update w_adjacency - we can either use frank wolfe or constant step size"""
        if self.step > 1000:
            # step_size_frank_wolfe = 2 / (self.step + 2)
            constant_step_size = 1/3
            s = torch.zeros(self.w_adjacency.shape, device=self.device)
            # s[torch.sum(self.w_lambda * differences, dim=-1) + norm_squared_differences*self.rho/2 - self.alpha < 0] = 1
            s[self.norm_squared_differences*self.rho/2 - self.alpha < 0] = 1
            self.w_adjacency += constant_step_size * (s - self.w_adjacency)
            # self.w_adjacency = s

        """logging and printing"""
        self.step += 1
        # loss_norm = 0
        # loss_inner_product = 0
        # loss_f = 0
        # for i in range(self.n_clients):
        #     for j in range(i+1, self.n_clients):
        #         loss_norm += self.rho*self.norm_squared_differences[i, j]/2 * self.w_adjacency[i, j]
        #         # loss_inner_product += torch.sum(self.w_lambda[i,j,:]*differences[i,j,:])*self.w_adjacency[i, j]
        #
        # for i in range(self.n_clients):
        #     for j, (p1, initial_p1) in enumerate(zip(train.models[i].parameters(), train.initial_models[i].parameters())):
        #         if train.shared_layers[j] == 1:
        #             loss_f += torch.sum(1/2*(p1.data - initial_p1.data)**2)
        #
        #     if self.step%50 == 0:
        #         print('loss_f: ', loss_f)

                # loss += self.rho*norm_squared_differences[i, j] + torch.sum(self.w_lambda[i,j,:]*differences[i,j,:])

        # loss = loss_inner_product + loss_norm + 1/self.n_clients*loss_f
        wandb.log({'w_sum' : torch.sum(self.w_adjacency), 'alpha' : self.alpha, 'max_grad_norm' : sum_smooth_grad, 'minmax grad': self.sum_grad,
                   'neighbor_distnces' : self.norm_squared_differences[0, 1]*self.rho/2})

        if self.step%50 == 0:
            # print('max_gradn:', self.max_grad)
            # print('grad_norms:', grad_norms)
            # print('param_norms*w: ', self.w_adjacency*torch.norm(stacked_params, p=2, dim=-1))
            print('differences:\n', self.rho*self.norm_squared_differences/2)
            # print('adjacency matrix:\n', self.w_adjacency)
            # print('inner product*w:', self.w_adjacency*torch.sum(self.w_lambda * differences, dim=-1))
            print('w:', self.w_adjacency)
            # print('s:', s)



    def unflat_lambda(self, flat_lambda, client, train):

        unflat_lambda = list()
        for i, p in enumerate(client.model.parameters()):
            if train.shared_layers[i] == 1:
                if flat_lambda is None:
                    unflat_lambda.append(torch.zeros(p.shape, device=self.device))
                else:
                    unflat_lambda.append(torch.reshape(flat_lambda[:p.numel()], p.shape))
                    flat_lambda = flat_lambda[p.numel():]

        return unflat_lambda

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

