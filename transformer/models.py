import math
import torch
import torch.nn as nn
import torch.nn.init as init


class MAB(nn.Module):

    def __init__(self, dim_Q, dim_K, dim_V, num_heads=6):
        super().__init__()
        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_Q = nn.Linear(dim_Q, dim_V)
        self.fc_K = nn.Linear(dim_K, dim_V)
        self.fc_V = nn.Linear(dim_K, dim_V)
        self.fc_O = nn.Linear(dim_V, dim_V)

    def forward(self, Q, V):
        B, _, _ = Q.shape
        Q, K, V = self.fc_Q(Q), self.fc_K(V), self.fc_V(V)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, dim=2), dim=0)
        K_ = torch.cat(K.split(dim_split, dim=2), dim=0)
        V_ = torch.cat(V.split(dim_split, dim=2), dim=0)
        A = torch.softmax(Q_ @ K_.transpose(1, 2) / self.dim_V ** 0.5, dim=2)
        O = torch.cat((Q_ + A @ V_).split(B,), dim=2)
        O = O + torch.relu(self.fc_O(O))
        return O


class SAB(nn.Module):

    def __init__(self, dim_in, dim_out, num_heads):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):

    def __init__(self, dim_in, dim_out, num_heads, num_inds):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads)
        init.xavier_uniform_(self.I)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):

    def __init__(self, dim, num_heads, num_seeds=1):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        self.mab = MAB(dim, dim, dim, num_heads)
        init.xavier_uniform_(self.S)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

