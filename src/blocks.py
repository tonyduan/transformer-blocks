import torch
import torch.nn as nn
import torch.nn.init as init


class MAB(nn.Module):
    """
    Multi-Head Attention Block [Vaswani et al. NeurIPS 2017].

    MAB(Q, V) = LN( O + ReLU(FC(O)) ), where O = LN( Q + MultiHeadAttn(Q, V) ).

    Notes
    -----
    We assume the last and penultimate dimensions correspond to the feature (i.e. embedding) and 
    token (i.e. words) dimensions respectively.
    """
    def __init__(self, dim_Q, dim_K, dim_V, num_heads=8, dropout_prob=0.1):
        super().__init__()
        self.dim_Q, self.dim_K, self.dim_V, self.num_heads = dim_Q, dim_K, dim_V, num_heads
        self.fc_Q = nn.Linear(dim_Q, dim_V)
        self.fc_K = nn.Linear(dim_K, dim_V)
        self.fc_V = nn.Linear(dim_V, dim_V)
        self.fc_O = nn.Linear(dim_V, dim_V)
        self.ln1 = nn.LayerNorm(dim_V)
        self.ln2 = nn.LayerNorm(dim_V)
        self.initialize_weights()

    def initialize_weights(self):
        pass

    def forward(self, Q, V):
        batch_size, _, _ = Q.shape
        Q, K, V = self.fc_Q(Q), self.fc_K(V), self.fc_V(V)
        split_size = self.dim_V // self.num_heads
        breakpoint()
        Q = torch.cat(Q.split(split_size, dim=-1), dim=0)
        K = torch.cat(K.split(split_size, dim=-1), dim=0)
        V = torch.cat(V.split(split_size, dim=-1), dim=0)
        A = torch.softmax(Q @ K.transpose(-1, -2) / self.dim_V ** 0.5, dim=-1)
        O = torch.cat((Q + A @ V).split(batch_size,), dim=-1)
        O = self.ln1(O)
        O = O + torch.relu(self.fc_O(O))
        O = self.ln2(O)
        return O


class SAB(nn.Module):
    """
    Self Attention Block [Lee et al. ICML 2019].

    Notes
    -----
    This is just a MAB block with the 
    """
    def __init__(self, dim_in, dim_out, num_heads):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    """
    Inducing Point Self-Attention Block [Lee et al. ICML 2019].
    """
    def __init__(self, dim_in, dim_out, num_heads, num_inds):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads)
        init.xavier_uniform_(self.I)

    def forward(self, X):
        H = self.mab0(self.I.repeat(len(X), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    """
    Pooling by Multi-Head Attention Block [Lee et al. ICML 2019].
    """
    def __init__(self, dim, num_heads, num_seeds=1):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        self.mab = MAB(dim, dim, dim, num_heads)
        init.xavier_uniform_(self.S)

    def forward(self, X):
        return self.mab(self.S.repeat(len(X), 1, 1), X)

