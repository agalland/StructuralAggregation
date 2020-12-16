import torch

import numpy as np
import torch.nn as nn

from torch import optim
from sklearn.metrics import accuracy_score, roc_auc_score
from model import SharedConv


class StructAgg(object):
    def __init__(self,
                 n_feat=None,
                 n_dim=None,
                 g_size=None,
                 bins=None,
                 path_save_weights=None,
                 num_layers=2,
                 out_dim=1,
                 lr=0.01,
                 num_epochs=100,
                 device="cpu"):

        self.g_size = g_size
        self.n_feat = n_feat
        self.n_dim = n_dim
        self.bins = bins
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.lr = lr
        self.path_save_weights = path_save_weights
        self.device = device
        self.accuracy_val = 0.
        self.reg = None
        self.class_weights_train = None

        self.criterion = None
        self.net = SharedConv(self.n_feat,
                              self.n_dim,
                              self.device,
                              self.bins,
                              self.out_dim,
                              self.num_layers).to(device)

    def fit(self,
            X_train,
            y_train,
            X_test,
            y_test,
            verbose=False,
            epoch_verbose=1,
            reg=False,
            alpha_reg=1.):
        self.reg = reg
        label_train = y_train.astype(np.int64)
        label_train = torch.LongTensor(label_train).to(self.device)
        label_test = y_test.astype(np.int64)
        label_test = torch.LongTensor(label_test).to(self.device)
        self.criterion = nn.NLLLoss()

        optimizer = optim.Adam(params=self.net.parameters(),
                               lr=self.lr,
                               weight_decay=1e-3)
        decrease_lr = False
        loss_test_min = np.infty
        loss_test = 0.
        loss_test_over = 0.
        loss_test_over_nb = 50
        for epoch in range(self.num_epochs):
            loss_test_prev = loss_test
            # Initialize grad
            optimizer.zero_grad()

            pred_train, loss_class_train = self.forward(X_train,
                                                        label_train)
            pred_test, loss_class_test = self.forward(X_test,
                                                      label_test)

            loss_train = pred_train
            loss_test = pred_test

            # Regularization
            if reg:
                loss_train += alpha_reg * loss_class_train
                loss_test += alpha_reg * loss_class_test

            # Loss backward
            loss_train.backward(retain_graph=True)
            optimizer.step()

            if decrease_lr:
                if torch.abs(loss_test - loss_test_prev) < 5e-4:
                    print("reduce lr")
                    self.lr /= 5.
                    optimizer = optim.Adam(params=self.net.parameters(),
                                           lr=self.lr,
                                           weight_decay=1e-4)
                    decrease_lr = False

            if loss_test.data < loss_test_min:
                loss_test_min = loss_test.data
                torch.save(self.net.state_dict(), self.path_save_weights)
                loss_test_over = 0
            else:
                loss_test_over += 1
            if loss_test_over > loss_test_over_nb:
                self.net.load_state_dict(torch.load(self.path_save_weights))
                print("break at epoch: {}".format(epoch-loss_test_over_nb-1))
                break

            if verbose and epoch % epoch_verbose == 0:
                print("Epoch: {}, loss train: {}, loss test: {},".format(epoch,
                                                                         loss_train.data,
                                                                         loss_test.data))

    def forward(self, X, y):
        pred = 0.
        p_class_loss = 0.
        for i, (adj, feature) in enumerate(X):
            adj_tensor = torch.Tensor(adj.todense()).to(self.device)
            feature_tensor = torch.Tensor(feature.todense()).to(self.device)
            output, p_class = self.net(feature_tensor, adj_tensor)
            if self.reg:
                p_class_loss += self.HLoss(p_class)
            output = output.view(1, -1)
            pred += self.criterion(output, y[i].unsqueeze(0))

        pred /= len(X)
        p_class_loss /= len(X)

        return pred, p_class_loss

    def get_class_select(self, X):
        p_classes = []
        for i, (adj, feature) in enumerate(X):
            adj_tensor = torch.Tensor(adj.todense()).to(self.device)
            feature_tensor = torch.Tensor(feature.todense()).to(self.device)
            output, p_class = self.net(feature_tensor, adj_tensor)
            p_classes.append(p_class.cpu().data.numpy())

        return p_classes

    def HLoss(self, prob):
        b = -prob * torch.log(prob + 1e-10)
        b = torch.sum(b, 1)
        b = torch.mean(b)

        return b

    def predict(self, X, y):
        pred = None
        for i, (adj, feature) in enumerate(X):
            adj_tensor = torch.Tensor(adj.todense()).to(self.device)
            feature_tensor = torch.Tensor(feature.todense()).to(self.device)
            output, p_class = self.net(feature_tensor, adj_tensor)
            output = output.view(1, -1)
            if pred is None:
                pred = output
            else:
                pred = torch.cat((pred, output), 0)
        pred_soft = pred.cpu().data.numpy()
        valpred, pred = torch.max(pred, 1)
        pred = pred.cpu().data.numpy()

        y_multi = np.zeros((len(y), len(np.unique(y))))
        y_multi[range(len(y)), y] = 1.
        acc = np.round(accuracy_score(y, pred), 3)
        roc = np.round(roc_auc_score(y_multi, pred_soft), 3)

        return acc, roc
