class grouping:
    def average_loss_grouping(self, clients, train):
        for client in clients:
            client.neighbors = list()

        for client1 in clients:
            for client2 in clients:
                if client1 != client2:
                    avg_model = train.find_average_parameters([client1.model, client2.model])
                    avg_acc, avg_loss = train.one_client_evaluation(avg_model, client1.dataset.val_loader)
                    acc1, loss1 = train.one_client_evaluation(client1.model, client1.dataset.val_loader)
                    acc2, loss2 = train.one_client_evaluation(client2.model, client2.dataset.val_loader)

                    if avg_loss < (loss1 + loss2)/2:
                        client1.neighbors.append(client2)
                        client2.neighbors.append(client1)

