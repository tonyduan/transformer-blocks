import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention [Vaswani et al. NeurIPS 2017].

    Scaled dot-product attention is performed over V, using K as keys and Q as queries.

        MultiHeadAttention(Q, V) = FC(SoftMax(1/√d QKᵀ) V) (concatenated over multiple heads),

    Notes
    -----
    (1) Q, K, V can be of different dimensions. Q and K are projected to dim_a and V to dim_o.
    (2) We assume the last and second last dimensions correspond to the feature (i.e. embedding)
        and token (i.e. words) dimensions respectively.
    """
    def __init__(self, dim_q, dim_k, dim_v, num_heads=8, dropout_prob=0.1, dim_a=None, dim_o=None, use_alibi=False):
        super().__init__()
        if dim_a is None:
            dim_a = dim_q
        if dim_o is None:
            dim_o = dim_q
        self.dim_a, self.dim_o, self.num_heads, self.use_alibi = dim_a, dim_o, num_heads, use_alibi
        self.fc_q = nn.Linear(dim_q, dim_a, bias=True)
        self.fc_k = nn.Linear(dim_k, dim_a, bias=True)
        self.fc_v = nn.Linear(dim_v, dim_o, bias=True)
        self.fc_o = nn.Linear(dim_o, dim_o, bias=True)
        self.dropout = nn.Dropout(dropout_prob)
        for module in (self.fc_q, self.fc_k, self.fc_v, self.fc_o):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0.)
        if self.use_alibi:
            log_slopes = torch.log(torch.arange(8, 0, -(8 / self.num_heads)))
            self.log_slopes = nn.Parameter(log_slopes.repeat(self.num_heads, 1, 1))

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
        q = torch.stack(q.split(self.dim_a // self.num_heads, dim=-1), dim=1)
        k = torch.stack(k.split(self.dim_a // self.num_heads, dim=-1), dim=1)
        v = torch.stack(v.split(self.dim_o // self.num_heads, dim=-1), dim=1)
        a = q @ k.transpose(-1, -2) / self.dim_a ** 0.5
        if self.use_alibi:
            arange = torch.arange(tsz, device=q.device)
            bias = -torch.abs(arange.unsqueeze(-1) - arange.unsqueeze(-2))
            bias = bias.repeat(self.num_heads, 1, 1) * torch.exp2(-torch.exp(self.log_slopes))
            a.add_(bias.unsqueeze(0))
        if mask is not None:
            assert mask.ndim in (2, 3)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            if mask.ndim == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            a.masked_fill_(mask == 0, -65504)
        a = self.dropout(torch.softmax(a, dim=-1))
        o = self.fc_o(a @ v).transpose(1, 2).flatten(2, 3)
        return o


class LinearAttention(nn.Module):
    """
    Linear Attention with ELU activation [Katharopoulos et al. 2020].

    Note the mask must either be causal or rectangular. Arbitrary masks are not supported.
    """
    def __init__(self, dim_q, dim_k, dim_v, num_heads=8, dropout_prob=0.1, dim_a=None, dim_o=None):
        super().__init__()
        if dim_a is None:
            dim_a = dim_q
        if dim_o is None:
            dim_o = dim_q
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
        bsz, tsz, _ = q.shape
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)
        q = torch.cat(q.split(self.dim_a // self.num_heads, dim=-1), dim=0)
        k = torch.cat(k.split(self.dim_a // self.num_heads, dim=-1), dim=0)
        v = torch.cat(v.split(self.dim_o // self.num_heads, dim=-1), dim=0)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        if mask is not None:
            assert mask.ndim in (2, 3)
            if mask.ndim == 3:
                assert torch.all(torch.tril(mask) == mask)
                o = torch.zeros_like(v)
                s = o.new_zeros(bsz)
                for i in range(tsz):
                    s += torch.linalg.vecdot(k[:, i], v[:, i], dim=-1)
                    o[:, i] = q[:, i] * s.unsqueeze(-1)
                o = self.fc_o(torch.cat(o.split(bsz, dim=0), dim=-1))
                return o
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
                k.masked_fill_(mask == 0, 0)
        context = torch.einsum("b t c, b t d -> b c d", k, v)
        numerator = torch.einsum("b t c, b c d -> b t d", q, context)
        denominator = torch.einsum("b t c, b c -> b t", q, torch.sum(k, dim=1))
        o = numerator / denominator.unsqueeze(-1)
        o = self.fc_o(torch.cat(o.split(bsz, dim=0), dim=-1))
        return o


class PointerAttention(nn.Module):
    """
    Pointer Attention [Vinyals et al. 2015].

    Note it returns *logits* which after softmax, will sum to 1 along the last dimension.
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
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderBlock(nn.Module):
    """
    Transformer encoder block [Vaswani et al. NeurIPS 2017].

    Note that this is the pre-LN version [Nguyen and Salazar 2019].
    """
    def __init__(self, dim, hidden_dim, num_heads=8, dropout_prob=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(dim, dim, dim, num_heads, dropout_prob)
        self.ffn = PositionwiseFFN(dim, hidden_dim, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

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
    def __init__(self, dim, hidden_dim, memory_dim=None, num_heads=8, dropout_prob=0.1):
        super().__init__()
        if memory_dim is None:
            memory_dim = dim
        self.attn = MultiHeadAttention(dim, dim, dim, num_heads, dropout_prob)
        self.mem_attn = MultiHeadAttention(dim, memory_dim, memory_dim, num_heads, dropout_prob)
        self.ffn = PositionwiseFFN(dim, hidden_dim, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)

    def forward(self, x, memory, mask=None, memory_mask=None):
        x_ = self.ln1(x)
        x = x + self.dropout(self.attn(x_, x_, x_, mask))
        x_ = self.ln2(x)
        x = x + self.dropout(self.mem_attn(x_, memory, memory, memory_mask))
        x_ = self.ln3(x)
        x = x + self.dropout(self.ffn(x_))
        return x


class CrossAttentionBlock(nn.Module):
    """
    Equivalent to a Transformer decoder block without the self-attention [Vaswani et al. 2017].

    Widely used in Perceiver IO architecture [Jaegle et al. 2022].

    Note that this is the pre-LN version [Nguyen and Salazar 2019].
    """
    def __init__(self, dim, hidden_dim, memory_dim=None, num_heads=8, dropout_prob=0.1):
        super().__init__()
        if memory_dim is None:
            memory_dim = dim
        self.mem_attn = MultiHeadAttention(dim, memory_dim, memory_dim, num_heads, dropout_prob)
        self.ffn = PositionwiseFFN(dim, hidden_dim, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x, memory, mask=None, memory_mask=None):
        x_ = self.ln1(x)
        x = x + self.dropout(self.mem_attn(x_, memory, memory, memory_mask))
        x_ = self.ln2(x)
        x = x + self.dropout(self.ffn(x_))
        return x


class InducedPointsEncoderBlock(nn.Module):
    """
    Induced self-attention block [Lee et al. 2019].

    Instead of O(N^2), this scales as O(NP) where P is the number of inducing points.
    """
    def __init__(self, dim, hidden_dim, num_heads, num_inds, dropout_prob=0.1):
        super().__init__()
        self.block1 = CrossAttentionBlock(dim, hidden_dim, num_heads, dropout_prob)
        self.block2 = CrossAttentionBlock(dim, hidden_dim, num_heads, dropout_prob)
        self.inducing_pts = nn.Parameter(torch.empty((num_inds, dim)))
        nn.init.xavier_normal_(self.inducing_pts)

    def forward(self, x):
        pts = self.inducing_pts.repeat(len(x), 1, 1)
        pts_transformed = self.block1(pts, x)
        x_transformed = self.block2(x, pts_transformed)
        return x_transformed


class PositionalEmbedding(nn.Module):
    """
    Positional Embedding module [Vaswani et al. NeurIPS 2017].

    Adds sinusoids with wavelengths of increasing length (lower freq) along the embedding dimension.
    First dimension has wavelength 2π while last dimension has wavelength max_length.
    """
    def __init__(self, dim, max_length=10000):
        super().__init__()
        self.emb = nn.Parameter(self.make_positional_embedding(dim, max_length))

    def forward(self, x):
        _, tsz, _ = x.shape
        return x + self.emb[:tsz, :]

    @staticmethod
    def make_positional_embedding(dim, max_length=10000):
        embedding = torch.zeros(max_length, dim)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(max_length / 2 / math.pi) / dim))
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)
        return embedding


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
