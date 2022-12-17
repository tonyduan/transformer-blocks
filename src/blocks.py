import math
import torch
import torch.nn as nn


class MultiHeadAttn(nn.Module):
    """
    Multi-Head Attention [Vaswani et al. NeurIPS 2017].

    Scaled dot-product attention is performed over V, using K as keys and Q as queries.

        MultiHeadAttn(Q, V) = FC(SoftMax(1/√d QKᵀ) V) (concatenated over multiple heads),

    Notes
    -----
    (1) Q, K, V can be of different dimensions. Q and K are projected to dim_a and V to dim_o.
    (2) We assume the last and second last dimensions correspond to the feature (i.e. embedding)
        and token (i.e. words) dimensions respectively.
    """
    def __init__(self, dim_q, dim_k, dim_v, num_heads=8, dropout_prob=0.1, dim_a=None, dim_o=None):
        super().__init__()
        if dim_a is None:
            dim_a = dim_q
        if dim_o is None:
            dim_o = dim_v
        self.dim_a, self.dim_o, self.num_heads = dim_a, dim_o, num_heads
        self.fc_q = nn.Linear(dim_q, dim_a, bias=True)
        self.fc_k = nn.Linear(dim_k, dim_a, bias=True)
        self.fc_v = nn.Linear(dim_v, dim_o, bias=True)
        self.fc_o = nn.Linear(dim_o, dim_o, bias=True)
        self.dropout = nn.Dropout(dropout_prob)
        for module in (self.fc_q, self.fc_k, self.fc_v, self.fc_o):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0.)

    def forward(self, q, k, v, mask=None):
        """
        Perform multi-head attention with given queries and values.

        Parameters
        ----------
        q: (bsz, tsz, dim_q)
        k: (bsz, tsz, dim_k)
        v: (bsz, tsz, dim_v)
        mask: (bsz, tsz) or (bsz, tsz, tsz), where 1 denotes keep and 0 denotes remove

        Returns
        -------
        O: (bsz, tsz, dim_o)
        """
        bsz, tsz, _ = q.shape
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)
        q = torch.cat(q.split(self.dim_a // self.num_heads, dim=-1), dim=0)
        k = torch.cat(k.split(self.dim_a // self.num_heads, dim=-1), dim=0)
        v = torch.cat(v.split(self.dim_o // self.num_heads, dim=-1), dim=0)
        a = q @ k.transpose(-1, -2) / self.dim_a ** 0.5
        if mask is not None:
            assert mask.ndim in (2, 3)
            if mask.ndim == 3:
                mask = mask.repeat(self.num_heads, 1, 1)
            if mask.ndim == 2:
                mask = mask.unsqueeze(-2).repeat(self.num_heads, tsz, 1)
            a.masked_fill_(mask == 0, -65504)
        a = self.dropout(torch.softmax(a, dim=-1))
        o = self.fc_o(torch.cat((a @ v).split(bsz, dim=0), dim=-1))
        return o


class PointerAttention(nn.Module):
    """
    Pointer Attention [Vinyals et al. 2015].

    Notes
    -----
    This class returns *logits* which after softmax, will sum to 1 along the last dimension.
    It's important that we use the function torch.log_softmax for numerical stability.
    """
    def __init__(self, dim_q, dim_k, dropout_prob=0.1, dim_a=None):
        super().__init__()
        if dim_a is None:
            dim_a = dim_q
        self.dim_a = dim_a
        self.fc_q = nn.Linear(dim_q, dim_a, bias=True)
        self.fc_k = nn.Linear(dim_k, dim_a, bias=True)
        for module in (self.fc_q, self.fc_k):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0.)

    def forward(self, q, k, mask=None):
        bsz, tsz, _ = q.shape
        q, k, = self.fc_q(q), self.fc_k(k)
        a = q @ k.transpose(-1, -2) / self.dim_a ** 0.5
        if mask is not None:
            assert mask.ndim in (2, 3)
            if mask.ndim == 3:
                mask = mask.repeat(self.num_heads, 1, 1)
            if mask.ndim == 2:
                mask = mask.unsqueeze(-2).repeat(self.num_heads, tsz, 1)
            a.masked_fill_(mask == 0, -65504)
        return torch.log_softmax(a, dim=-1)


class PositionwiseFFN(nn.Module):
    """
    Position-wise FFN [Vaswani et al. NeurIPS 2017].
    """
    def __init__(self, dim, hidden_dim, dropout_prob=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout_prob)
        for module in (self.fc1, self.fc2):
            nn.init.kaiming_normal_(module.weight)
            nn.init.constant_(module.bias, 0.)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


class EncoderBlock(nn.Module):
    """
    Transformer encoder block [Vaswani et al. NeurIPS 2017].

    Note that this is the pre-LN version [Nguyen and Salazar 2019].
    """
    def __init__(self, dim, hidden_dim, num_heads=8, dropout_prob=0.1):
        super().__init__()
        self.attn = MultiHeadAttn(dim, dim, dim, num_heads, dropout_prob)
        self.ffn = PositionwiseFFN(dim, hidden_dim, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.ln1 = ScaleNorm(dim)
        self.ln2 = ScaleNorm(dim)

    def forward(self, x, mask=None):
        x_ = self.ln1(x)
        x = x + self.dropout(self.attn(x_, x_, x_, mask))
        x_ = self.ln2(x)
        x = x + self.dropout(self.ffn(x_))
        return x


class DecoderBlock(nn.Module):
    """
    Transformer decoder block [Vaswani et al. 2017].

    Note that this is the pre-LN version [Nguyen and Salazar 2019].
    """
    def __init__(self, dim, hidden_dim, num_heads=8, dropout_prob=0.1):
        super().__init__()
        self.attn = MultiHeadAttn(dim, dim, dim, num_heads, dropout_prob)
        self.mem_attn = MultiHeadAttn(dim, dim, dim, num_heads, dropout_prob)
        self.ffn = PositionwiseFFN(dim, hidden_dim, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.ln1 = ScaleNorm(dim)
        self.ln2 = ScaleNorm(dim)
        self.ln3 = ScaleNorm(dim)

    def forward(self, x, memory, mask=None, memory_mask=None):
        x_ = self.ln1(x)
        x = x + self.dropout(self.attn(x_, x_, x_, mask))
        x_ = self.ln2(x)
        x = x + self.dropout(self.mem_attn(x_, memory, memory, memory_mask))
        x_ = self.ln3(x)
        x = x + self.dropout(self.ffn(x_))
        return x


class ScaleNorm(nn.Module):
    """
    ScaleNorm [Nguyen and Salazar 2019].
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1) * dim ** 0.5)
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        x = x / n * self.g
        return x


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module [Vaswani et al. NeurIPS 2017].

    Adds sinusoids with wavelengths of increasing length (lower freq) along the embedding dimension.
    First dimension has wavelength 2π while last dimension has wavelength max_length.
    """
    def __init__(self, dim, dropout_prob=0.0, max_length=10000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        encoding = torch.zeros(max_length, dim)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(max_length / 2 / math.pi) / dim))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding)

    def forward(self, x):
        _, tsz, _ = x.shape
        return self.dropout(x + self.encoding[:tsz, :])


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
        self.ln1 = ScaleNorm(dim) if use_layer_norm else nn.Identity()
        self.ln2 = ScaleNorm(dim) if use_layer_norm else nn.Identity()
        self.fc = nn.Linear(dim, dim, bias=True)
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
