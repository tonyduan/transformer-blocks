import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import einsum, rearrange, reduce, repeat


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
        self.scale = self.dim_a ** -0.5
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
        q = rearrange(q, "b t (g c) -> b g t c", g=self.num_heads)
        k = rearrange(k, "b t (g c) -> b g t c", g=self.num_heads)
        v = rearrange(v, "b t (g c) -> b g t c", g=self.num_heads)
        a = einsum(q, k, "b g q c, b g k c -> b g q k") * self.scale
        if mask is not None:
            assert mask.ndim in (2, 3)
            if mask.ndim == 3:
                mask = rearrange(mask, "b q k -> b 1 q k")
            if mask.ndim == 2:
                mask = rearrange(mask, "b k -> b 1 1 k")
            a.masked_fill_(mask == 0, -65504)
        a = self.dropout(F.softmax(a, dim=-1))
        o = einsum(a, v, "b g q k, b g k c -> b g q c")
        o = rearrange(o, "b g t c -> b t (g c)")
        o = self.fc_o(o)
        return o


class LinearAttention(nn.Module):
    """
    Linear Attention with ELU activation [Katharopoulos et al. 2020].

    Note the mask must be rectangular. Arbitrary masks are not supported.
    Causal mask is doable not implemented here but can be unrolled as an RNN step (see paper).
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
        bsz, *_ = q.shape
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        q = rearrange(q, "b t (g c) -> b g t c", g=self.num_heads)
        k = rearrange(k, "b t (g c) -> b g t c", g=self.num_heads)
        v = rearrange(v, "b t (g c) -> b g t c", g=self.num_heads)
        if mask is not None:
            assert mask.ndim == 2
            mask = rearrange(mask, "b k -> b 1 1 k")
            k.masked_fill_(mask == 0, 0)
        # These can be cached, they take the sum over tokens
        k_sum = reduce(k, "b g t c -> b g c", "sum")
        k_v = einsum(k, v, "b g t c_k, b g t c_v -> b g c_k c_v")
        # These need to be recomputed in causal rollout
        q_k_v = einsum(q, k_v, "b g q c_k, b g c_k c_v -> b g q c_v")
        q_k_sum = einsum(q, k_sum, "b g q c, b g c -> b g q")
        # This division is safe because the activation is ELU(x) + 1
        o = q_k_v / q_k_sum.unsqueeze(-1)
        o = rearrange(o, "b g t c -> b t (g c)")
        o = self.fc_o(o)
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
        self.scale = self.dim_a ** -0.5
        self.fc_q = nn.Linear(dim_q, dim_a, bias=True)
        self.fc_k = nn.Linear(dim_k, dim_a, bias=True)
        for module in (self.fc_q, self.fc_k):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0.)

    def forward(self, q, k, mask=None):
        bsz, *_ = q.shape
        q, k, = self.fc_q(q), self.fc_k(k)
        a = torch.einsum("b q c, b k c -> b q k", q, k) * self.scale
        if mask is not None:
            assert mask.ndim in (2, 3)
            if mask.ndim == 2:
                mask = rearrange(mask, "b k -> b 1 k")
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
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0.)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderBlock(nn.Module):
    """
    Transformer encoder block [Vaswani et al. NeurIPS 2017].
    """
    def __init__(self, dim, hidden_dim, num_heads=8, dropout_prob=0.1):
        super().__init__()
        self.attn = LinearAttention(dim, dim, dim, num_heads, dropout_prob)
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
        self.latent_pts = nn.Parameter(torch.empty((1, num_inds, dim)))
        nn.init.xavier_normal_(self.inducing_pts)

    def forward(self, x):
        bsz, *_ = x.shape
        latent_pts = repeat(self.latent_pts, "1 t c -> b t c", b=bsz)
        latent_pts_to_x = self.block1(latent_pts, x)
        x_to_latent_pts = self.block2(x, latent_pts_to_x)
        return x_to_latent_pts


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


class RMSNorm(nn.Module):
    """
    RMSNorm [Zhang and Sennich 2019].
    """
    def __init__(self, dim, eps=1e-5):
        super().__init_()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        x = x / n * self.scale * self.dim ** -0.5
        return x
