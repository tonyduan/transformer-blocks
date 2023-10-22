from argparse import ArgumentParser
import logging

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.blocks import EncoderBlock, RMSNorm


def gen_data(m, n, k, include_start_token=True):
    """
    Returns m batches of n tokens, see below where START = 0 and RAND ~ (1, ..., k - 1).
        [[START, RAND, RAND, ..., RAND],
         [START, RAND, RAND, ..., RAND]]
    """
    if include_start_token:
        x = np.random.randint(k - 1, size=(m, n - 1)) + 1
        x = np.c_[np.zeros((m, 1), dtype=np.int32), x]
    else:
        x = np.random.randint(k, size=(m, n))
    y = np.vstack([np.arange(n) + 1 for _ in range(m)])
    return x, y


class CountExtractor(nn.Module):

    def __init__(self, vocab_size, dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=dim)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(dim, 4 * dim, num_heads=1),
            EncoderBlock(dim, 4 * dim, num_heads=1),
        ])
        self.ln = RMSNorm(dim)
        self.out_conv = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1, bias=True),
        )
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, x):
        bsz, tsz = x.shape
        causal_mask = torch.tril(torch.ones((tsz, tsz), dtype=torch.int32)).unsqueeze(dim=0)
        x = self.embedding(x)
        for block in self.encoder_blocks:
            x = block(x, mask=causal_mask)
        x = self.ln(x)
        y_hat = self.out_conv(x).squeeze(dim=-1)
        return y_hat


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--iterations", type=int, default=300)
    argparser.add_argument("--vocab-size", type=int, default=20)
    argparser.add_argument("--n", type=int, default=100)
    argparser.add_argument("--dim", type=int, default=20)
    argparser.add_argument("--include-start-token", type=int, default=0)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model = CountExtractor(vocab_size=args.vocab_size, dim=args.dim)
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iterations)

    x_tr, y_tr = gen_data(500, n=args.n, k=args.vocab_size,
                          include_start_token=args.include_start_token)
    x_tr = torch.tensor(x_tr, dtype=torch.int32)
    y_tr = torch.tensor(y_tr, dtype=torch.int32)

    for i in range(args.iterations):

        optimizer.zero_grad()
        y_hat = model.forward(x_tr)
        loss = 0.5 * ((y_tr - y_hat) ** 2)
        loss.mean().backward()
        optimizer.step()
        scheduler.step()
        if i % 5 == 0:
            logger.info(f"Iter: {i}\t"
                        f"Loss: {loss.mean().data:.2f}\t")

    model.eval()
    x_te, y_te = gen_data(100, n=args.n, k=args.vocab_size,
                          include_start_token=args.include_start_token)
    x_te = torch.tensor(x_te, dtype=torch.int32)
    y_te = torch.tensor(y_te, dtype=torch.int32)
    with torch.no_grad():
        y_hat = model.forward(x_te)

    y_te = y_te.data.numpy()
    y_hat = y_hat.data.numpy()

    r2_score = 1 - np.sum((y_te - y_hat) ** 2) / np.sum((y_te - y_te.mean()) ** 2)
    l1_error = np.abs(y_hat - y_te).mean()
    logger.info(f"R2 Score: {r2_score}")
    logger.info(f"L1 Error: {l1_error}")

    NUM_SAMPLES_TO_VIS = 10

    x_axis = np.arange(args.n) + 1
    plt.figure(figsize=(8, 8))
    plt.plot(x_axis, x_axis, "-.", color="grey")
    for i in range(NUM_SAMPLES_TO_VIS):
        plt.plot(y_hat[i], ".", color="black")

    plt.tight_layout()
    plt.savefig("./examples/count.png")
    plt.show()
