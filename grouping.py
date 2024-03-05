from Models.smallnet import SmallNet
import torch
import wandb
import matplotlib.pyplot as plt


class Grouping:

    def __init__(self, n_clients, alpha=None, rho=None, w_adjacency=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_clients = n_clients
        if w_adjacency is None:
            self.w_adjacency = torch.ones((self.n_clients, self.n_clients), device=self.device)
        else:
            self.w_adjacency = w_adjacency

        # self.w_adjacency = torch.eye(self.n_clients, device=self.device)
        # a = [1,1,1,1,0,0,0,0]
        # b = [0,0,0,0,1,1,1,1]
        # self.w_adjacency = torch.tensor([a, a, a, a, b, b, b, b], device=self.device)
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

        """alpha scheduler"""
        stacked_gradients = torch.stack(flattened_grads_list)
        grad_norms = torch.norm(stacked_gradients, p=2, dim=-1)

        stacked_params = torch.stack(flattened_params_list)
        self.differences = stacked_params[:, None, :] - stacked_params[None, :, :]
        self.next_step_differences = (
                    (stacked_params[:, None, :] - train.learning_rate * stacked_gradients[:, None, :]) -
                    (stacked_params[None, :, :] - train.learning_rate * stacked_gradients[None, :, :]))
        self.norm_squared_differences = torch.norm(self.differences, p=2, dim=-1) ** 2
        self.next_step_norm_squared_differences = torch.norm(self.next_step_differences, p=2, dim=-1) ** 2

        """update w_adjacency"""

        self.step += 1
        applied_alpha = self.alpha
        if self.step > 30000:
            applied_alpha = self.alpha / self.step
        s = torch.zeros(self.w_adjacency.shape, device=self.device)
        s[self.norm_squared_differences * self.rho / 2 - applied_alpha < 0] = 1

        self.w_adjacency = (1 - self.beta) * self.w_adjacency + self.beta * s

        """logging and printing"""

        if train.master_process and self.step % 50 == 0:
            print('w:', self.w_adjacency)
            print('differences:\n', self.rho * self.norm_squared_differences / 2)

            plt.imshow(self.w_adjacency.cpu().numpy(), cmap='viridis', interpolation='none', vmin=0, vmax=1)
            wandb.log({'w_sum': torch.sum(self.w_adjacency), 'alpha': applied_alpha,
                       'neighbor_distnces': self.norm_squared_differences[0, 1] * self.rho / 2,
                       'non-neighbor_distance': self.norm_squared_differences[0, -1] * self.rho / 2,
                       'adjacency matrix': wandb.Image(plt)})
