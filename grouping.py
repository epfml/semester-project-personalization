from model import Net
import math
class Grouping:
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

