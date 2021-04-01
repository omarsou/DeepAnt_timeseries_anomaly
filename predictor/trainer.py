import torch
import numpy as np


class DeepAntTrainer:
    def __init__(self, model, optimizer, scheduler, loss, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.device = device

    def fit(self, num_epochs, train_loader, test_loader, print_freq=10):
        self.model.fitted = True
        for epoch in range(num_epochs):
            self.model.train()
            train_average_loss = 0.0
            for i, (x, y) in enumerate(train_loader):
                self.optimizer.zero_grad()

                x, y = x.to(self.device), y.to(self.device)

                y_pred = self.model(x.float())
                loss = self.loss(y_pred, y)

                loss.backward()
                self.optimizer.step()

                train_average_loss += loss.item()

            train_average_loss = train_average_loss / len(train_loader)

            self.model.eval()
            test_average_loss = 0.0
            for i, (x, y) in enumerate(test_loader):
                with torch.no_grad():
                    x, y = x.to(self.device), y.to(self.device)

                    y_pred = self.model(x.float())
                    loss = self.loss(y_pred, y)

                    test_average_loss += loss.item()

            test_average_loss = test_average_loss / len(test_loader)
            self.scheduler.step(test_average_loss)

            if epoch % print_freq == 0:
                print('Epoch [{}/{}], TrainLoss: {:.4f}, TestLoss: {:.4f}, Learning rate = {}'.format(epoch + 1,
                                                                                                      num_epochs,
                                                                                                      train_average_loss,
                                                                                                      test_average_loss,
                                                                                                      self.optimizer.param_groups[
                                                                                                          0]['lr']))

    def predict(self, test_loader, num_samples, batch_size):
        self.model.eval()
        predictions = np.zeros((num_samples, self.model.output_dim))
        for i, (x, y) in enumerate(test_loader):
            x = x.to(self.device)
            if i != len(test_loader) - 1:
                predictions[i * batch_size: (i + 1) * batch_size] = self.model(x.float()).detach().cpu().numpy()
            else:
                last_pred = self.model(x.float()).detach().cpu().numpy()
                predictions[i * batch_size: i * batch_size + last_pred.shape[0]] = last_pred
        return predictions