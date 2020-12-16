import torch

import torch.nn as nn
import torch.nn.functional as F

from layer import GraphConvolution


class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class SharedConv(nn.Module):
    def __init__(self, nfeat, ndim, device, bins=10, out_dim=1, num_layers=2):
        """Dense version of GAT."""
        super(SharedConv, self).__init__()
        self.nfeat = nfeat
        self.bins = bins
        self.out_dim = out_dim
        self.device = device
        self.ndim = ndim
        self.num_layers = num_layers

        self.dimPred = self.bins * self.ndim

        self.add_module("conv_0", GraphConvolution(self.nfeat, self.ndim))
        for k in range(1, num_layers):
            self.add_module("conv_" + str(k), GraphConvolution(self.ndim,
                                                               self.ndim))

        self.p = nn.Parameter(torch.zeros(size=(self.ndim*self.num_layers,
                                                self.bins)))
        nn.init.xavier_uniform_(self.p.data, gain=1.414)

        self.fc1 = nn.Linear(self.dimPred, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, self.out_dim)

        self.gcns = AttrProxy(self, 'conv_')

    def forward(self, x, adj):
        xs = []
        # First layer of GCN
        x = self.gcns[0](x, adj)
        x = F.relu(x)
        x = F.normalize(x, p=2, dim=1)

        xs.append(x)

        for k in range(1, self.num_layers):
            # Other layers of GCN
            x1 = x
            x = self.gcns[k](x, adj)
            x = F.relu(x)
            x = F.normalize(x+x1, p=2, dim=1)
            xs.append(x)

        # Compute assignment matrix
        c = self.cluster_mean(xs, self.p)

        x = torch.matmul(c, x)
        x = x.view(1, -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.log_softmax(x, 1)

        return x, c

    def cluster_mean(self, xs, p):
        x = xs[0]
        if len(xs) > 1:
            for x_i in xs[1:]:
                x = torch.cat((x, x_i), 1)
        c = torch.matmul(x, p)
        c = torch.softmax(c, 1)
        c = c.transpose(0, 1)

        return c
