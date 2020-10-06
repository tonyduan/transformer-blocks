import torch
import torch.nn as nn


class MultiHeadAttn(nn.Module):
    """
    Multi-Head Attention [Vaswani et al. NeurIPS 2017].

    Scaled dot-product attention is performed over V, using K as keys and Q as queries.

        MultiHeadAttn(Q, V) = FC(SoftMax(1/√d QKᵀ) V) (concatenated over multiple heads),

    Notes
    -----
    (1) Q, K, V can be of different dimensions, though Q/K are projected to the dimensionality of V.
    (2) We assume the last and second last dimensions correspond to the feature (i.e. embedding)
        and token (i.e. words) dimensions respectively.
    """
    def __init__(self, dim_q, dim_k, dim_v, num_heads=8, dropout_prob=0.1):
        super().__init__()
        self.dim_q, self.dim_v, self.num_heads = dim_q, dim_v, num_heads
        self.fc_q = nn.Linear(dim_q, dim_v, bias=True)
        self.fc_k = nn.Linear(dim_k, dim_v, bias=True)
        self.fc_v = nn.Linear(dim_v, dim_v, bias=True)
        self.fc_o = nn.Linear(dim_v, dim_v, bias=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.initialize_weights()

    def initialize_weights(self):
        for module in (self.fc_q, self.fc_k, self.fc_v, self.fc_o):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0.)

    def forward(self, q, k, v, mask=None):
        """
        Perform multi-head attention with given queries and values.

        Parameters
        ----------
        q: (batch_size, token_size, dim_q)
        k: (batch_size, token_size, dim_q)
        v: (batch_size, token_size, dim_v)
        mask: (batch_size, token_size), where 1 denotes keep and 0 denotes remove

        Returns
        -------
        O: (batch_size, token_size, dim_v)
        """
        batch_size = len(q)
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)
        split_size = self.dim_v // self.num_heads
        q = torch.cat(q.split(split_size, dim=-1), dim=0)
        k = torch.cat(k.split(split_size, dim=-1), dim=0)
        v = torch.cat(v.split(split_size, dim=-1), dim=0)
        a = q @ k.transpose(-1, -2) / self.dim_v ** 0.5
        if mask is not None:
            a[mask.unsqueeze(-2) == 0] = -65504
        a = self.dropout(torch.softmax(a, dim=-1))
        o = self.fc_o(torch.cat((a @ v).split(batch_size, dim=0), dim=-1))
        return o


class PositionwiseFFN(nn.Module):
    """
    Position-wise FFN [Vaswani et al. NeurIPS 2017].
    """
    def __init__(self, dim, hidden_dim, dropout_prob=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.initialize_weights()

    def initialize_weights(self):
        for module in (self.fc1, self.fc2):
            nn.init.kaiming_normal_(module.weight)
            nn.init.constant_(module.bias, 0.)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


class EncoderBlock(nn.Module):
    """
    Transformer encoder block [Vaswani et al. NeurIPS 2017].
    """
    def __init__(self, dim, hidden_dim, num_heads=8, dropout_prob=0.1):
        super().__init__()
        self.attn = MultiHeadAttn(dim, dim, dim, num_heads, dropout_prob)
        self.ffn = PositionwiseFFN(dim, hidden_dim, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.ln1(x + self.dropout(self.attn(x, x, x, mask)))
        x = self.ln2(x + self.dropout(self.ffn(x)))
        return x


class DecoderBlock(nn.Module):
    """
    Transformer decoder block [Vaswani et al. 2017].
    """
    def __init__(self, dim, hidden_dim, num_heads=8, dropout_prob=0.1):
        super().__init__()
        self.attn = MultiHeadAttn(dim, dim, dim, num_heads, dropout_prob)
        self.mem_attn = MultiHeadAttn(dim, dim, dim, num_heads, dropout_prob)
        self.ffn = PositionwiseFFN(dim, hidden_dim, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)

    def forward(self, x, memory, mask=None, memory_mask=None):
        x = self.ln1(x + self.dropout(self.attn(x, x, x, mask)))
        x = self.ln2(x + self.dropout(self.mem_attn(x, memory, memory, memory_mask)))
        x = self.ln3(x + self.dropout(self.ffn(x)))
        return x


class MAB(nn.Module):
    """
    Multi-Head Attention Block [Lee et al. ICML 2019].

    Notes
    -----
    Here the queries Q and values V must be of the same dimension. We use values V as keys K.
    """
    def __init__(self, dim, num_heads, use_layer_norm=False):
        super().__init__()
        self.attn = MultiHeadAttn(dim, dim, dim, num_heads, dropout_prob=0.)
        self.ln1 = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.ln2 = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.fc = nn.Linear(dim, dim, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_out")
        nn.init.constant_(self.fc.bias, 0.)

    def forward(self, q, v):
        out = self.ln1(q + self.attn(q, v, v))
        return self.ln2(out + self.fc(out))


class SAB(nn.Module):
    """
    Self Attention Block [Lee et al. ICML 2019].
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.mab = MAB(dim, num_heads)

    def forward(self, x):
        return self.mab(x, x)


class ISAB(nn.Module):
    """
    Induced Self-Attention Block [Lee et al. ICML 2019].

    Notes
    -----
    Instead of self-attention between all queries and values (N^2), we use M induced points (MN).
    """
    def __init__(self, dim, num_heads, num_inds):
        super().__init__()
        self.inducing_pts = nn.Parameter(torch.Tensor(1, num_inds, dim))
        self.mab1 = MAB(dim, num_heads)
        self.mab2 = MAB(dim, num_heads)
        nn.init.xavier_normal_(self.inducing_pts)

    def forward(self, x):
        transformed_pts = self.mab1(self.inducing_pts.repeat(len(x), 1, 1), x)
        return self.mab2(x, transformed_pts)


class PMA(nn.Module):
    """
    Pooling by Multi-Head Attention Block [Lee et al. ICML 2019].
    """
    def __init__(self, dim, num_heads, num_seeds=1):
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        self.mab = MAB(dim, num_heads)
        nn.init.xavier_normal_(self.seed_vectors)

    def forward(self, x):
        return self.mab(self.seed_vectors.repeat(len(x), 1, 1), x)

