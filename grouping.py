from Models.smallnet import SmallNet
import torch
import wandb
import matplotlib.pyplot as plt

class Grouping:

    def __init__(self, n_clients, lr_lambda=None, alpha=None, rho=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_clients = n_clients
        self.w_adjacency = torch.ones((self.n_clients, self.n_clients), device=self.device)
        # self.w_adjacency = torch.eye(self.n_clients, device=self.device)
        # a = [1,1,1,1,0,0,0,0]
        # b = [0,0,0,0,1,1,1,1]
        # self.w_adjacency = torch.tensor([a, a, a, a, b, b, b, b], device=self.device)
        self.w_lambda = None
        self.lr_lambda = lr_lambda
        self.step = 0
        self.alpha = alpha
        self.rho = rho
        self.smooth_grads = torch.zeros(self.n_clients, device=self.device)
        self.patience = torch.zeros(self.n_clients, device=self.device)
        self.differences = None
        self.prv = 0
        self.norm_squared_differences = None
        self.alpha_encourage = torch.zeros(self.n_clients, device=self.device)
        self.beta = 0.1
        self.right_grouping = 0
        self.wrong_edges_right_number = 0
        self.wrong_number = 0
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

        # smooth_grad_norms = grad_norms
        #
        # if self.step != 0:
        #     smooth_grad_norms = 0.99*self.prv + 0.01*grad_norms
        #
        # self.prv = smooth_grad_norms
        #
        # # if self.step % 1000 == 0:
        # #     self.alpha *= 0.9
        #
        # if self.step >= 500:
        #     if self.step == 500:
        #         # self.smooth_grads = smooth_grad_norms
        #         self.smooth_grads = train.train_losses
        #     else:
        #         # reduced_grads = self.smooth_grads > smooth_grad_norms
        #         reduced_grads = self.smooth_grads > train.train_losses
        #         self.patience[reduced_grads] = 0
        #         self.alpha_encourage[reduced_grads] += 1
        #
        #         self.patience[~reduced_grads] += 1
        #
        #         if torch.amax(self.patience) >= 200:
        #             self.alpha_encourage = torch.zeros(self.n_clients, device=self.device)
        #             self.alpha *= 0.9
        #             self.patience = torch.zeros(self.n_clients, device=self.device)
        #             print('alpha redueced! current alpha:', self.alpha)
        #         elif torch.amin(self.alpha_encourage) > 0:
        #             self.alpha *= 1.1
        #             self.alpha_encourage = torch.zeros(self.n_clients, device=self.device)
        #             self.patience = torch.zeros(self.n_clients, device=self.device)
        #             print('alpha increased! current alpha:', self.alpha)
        #
        #         # self.smooth_grads[reduced_grads] = smooth_grad_norms[reduced_grads]
        #         self.smooth_grads[reduced_grads] = train.train_losses[reduced_grads]
            # if self.sum_grad is None or self.sum_grad > sum_smooth_grad:
            #     self.sum_grad = sum_smooth_grad
            #     self.patience = 0
            #     self.alpha_encourage += 1
            #     if self.alpha_encourage == 50:
            #         self.alpha *= 1.1
            #         self.alpha_encourage = 0
            #         print('alpha increased! current alpha:', self.alpha)
            # else:
            #     self.patience += 1
            #
            # if self.patience == 500:
            #     self.alpha_encourage = 0
            #     self.alpha *= 0.9
            #     self.patience = 0
            #     print('alpha redueced! current alpha:', self.alpha)



        stacked_params = torch.stack(flattened_params_list)
        self.differences = stacked_params[:, None, :] - stacked_params[None, :, :]
        self.next_step_differences = ((stacked_params[:, None, :] - train.learning_rate * stacked_gradients[:, None, :]) -
                                      (stacked_params[None, :, :] - train.learning_rate * stacked_gradients[None, :, :]))
        self.norm_squared_differences = torch.norm(self.differences, p=2, dim = -1)**2
        self.next_step_norm_squared_differences = torch.norm(self.next_step_differences, p=2, dim = -1)**2

        # self.alpha = min(self.alpha, torch.amax(self.norm_squared_differences)*self.rho)
        # self.norm_squared_differences.fill_diagonal_(float('inf'))
        # self.alpha = max(self.alpha, torch.amin(self.norm_squared_differences)*self.rho/4)
        # self.norm_squared_differences.fill_diagonal_(0)

        # self.alpha = 0.05
        """update w_lambda"""

        # if self.w_lambda is None:
        #     self.w_lambda = torch.zeros((self.n_clients, self.n_clients, stacked_params.shape[1]), device=self.device)
        #
        # self.w_lambda += self.lr_lambda * (self.w_adjacency[:, :, None] * differences)

        """update w_adjacency - we can either use frank wolfe or constant step size"""


        # self.w_adjacency[self.norm_squared_differences >= self.next_step_norm_squared_differences] = 1
        # self.w_adjacency[self.norm_squared_differences < self.next_step_norm_squared_differences] = 0

        # if torch.equal(self.w_adjacency, torch.tensor([[1, 1, 0], [1, 1, 0], [0, 0 , 1]], device=self.device)):
        #     self.right_grouping += 1
        # elif torch.sum(self.w_adjacency) == 5:
        #     self.wrong_edges_right_number += 1
        # else:
        #     self.wrong_number += 1

        # if self.step > 1000:
            # step_size_frank_wolfe = 2 / (self.step + 2)
            # constant_step_size = 1/3
            # s = torch.zeros(self.w_adjacency.shape, device=self.device)
            # s[torch.sum(self.w_lambda * differences, dim=-1) + norm_squared_differences*self.rho/2 - self.alpha < 0] = 1
            # s[self.norm_squared_differences*self.rho/2 - self.alpha < 0] = 1
            # self.w_adjacency += constant_step_size * (s - self.w_adjacency)
            # self.w_adjacency = s
        self.step += 1

        s = torch.zeros(self.w_adjacency.shape, device=self.device)
        s[self.norm_squared_differences*self.rho/2 - self.alpha/self.step < 0] = 1

        self.w_adjacency = (1 - self.beta)*self.w_adjacency + self.beta*s


        """logging and printing"""

        # loss_norm = 0
        # loss_inner_product = 0
        loss_f = 0
        # for i in range(self.n_clients):
        #     for j in range(i+1, self.n_clients):
        #         loss_norm += self.rho*self.norm_squared_differences[i, j]/2 * self.w_adjacency[i, j]
        #         # loss_inner_product += torch.sum(self.w_lambda[i,j,:]*differences[i,j,:])*self.w_adjacency[i, j]
        #


        if self.step%50 == 0:
            # for i in range(self.n_clients):
            #     for j, (p1, initial_p1) in enumerate(
            #             zip(train.models[i].parameters(), train.initial_models[i].parameters())):
            #         if train.shared_layers[j] == 1:
            #             loss_f += self.w_adjacency[i, j] * torch.sum(1 / 2 * (p1.data - initial_p1.data) ** 2)
            #
            # print('loss_f: ', loss_f)
            print('w:', self.w_adjacency)

                # loss += self.rho*norm_squared_differences[i, j] + torch.sum(self.w_lambda[i,j,:]*differences[i,j,:])

        # loss = loss_inner_product + loss_norm + 1/self.n_clients*loss_f
        w_sum = self.right_grouping + self.wrong_edges_right_number + self.wrong_number


        wandb.log({'w_sum': torch.sum(self.w_adjacency), 'alpha' : self.alpha/self.step,
                   # 'smooth_grad0' : smooth_grad_norms[0], 'smooth_grad2' : smooth_grad_norms[2], 'min_grad2': self.smooth_grads[2],
                   'neighbor_distnces' : self.norm_squared_differences[0, 1]*self.rho/2,
                   'non-neighbor_distance' : self.norm_squared_differences[0, -1]*self.rho/2,
                   'min_encourage': torch.amin(self.alpha_encourage)})

        if self.step%50 == 0:
            plt.imshow(self.w_adjacency.cpu().numpy(), cmap='viridis', interpolation='none')
            wandb.log({'adjacency matrix': wandb.Image(plt)})
            # print('max_gradn:', self.max_grad)
            # print('grad_norms:', grad_norms)
            # print('param_norms*w: ', self.w_adjacency*torch.norm(stacked_params, p=2, dim=-1))
            print('differences:\n', self.rho*self.norm_squared_differences/2)
            # print('adjacency matrix:\n', self.w_adjacency)
            # print('inner product*w:', self.w_adjacency*torch.sum(self.w_lambda * differences, dim=-1))

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
                    avg_model = SmallNet()
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

