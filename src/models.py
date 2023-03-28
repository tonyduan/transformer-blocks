import torch.nn as nn
import torch.nn.functional as F
from src.blocks import EncoderBlock, PositionalEncoding, PMA


class BERTLanguageModel(nn.Module):
    """
    BERT-based [Devlin et al. NAACL 2019] language model to predict the next word given a context.

    This is just a stack of encoder blocks followed by a pooling layer for classification.

    Notes
    -----
    Instead of a <CLS> token, we use a pooling by multi-head attention (PMA) block for final layer.
    """
    def __init__(self, device, vocab_size, num_layers, dim, hidden_dim,
                 num_heads=8, dropout_prob=0.1, max_length=10000):
        super().__init__()
        self.device = device
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=1)
        self.positional_encoding = PositionalEncoding(dim, dropout_prob, max_length)
        self.pool = PMA(dim, num_heads, num_seeds=1)
        self.fc = nn.Linear(dim, vocab_size, bias=True)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(EncoderBlock(dim, hidden_dim, num_heads, dropout_prob))
        self.initialize_weights()
        self.set_device()

    def initialize_weights(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def set_device(self):
        for m in self.modules():
            m = m.to(self.device)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x).squeeze()
        x = self.fc(x)
        return x

    def loss(self, x, y):
        logits = self.forward(x)
        return F.cross_entropy(logits, y, reduction="none")

