from argparse import ArgumentParser
import logging

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.blocks import EncoderBlock, PointerAttention, RMSNorm


def gen_data(m, n, dim=2):
    """
    m batches of n samples each ~ N, and of dimension 2.
    """
    x = np.random.randn(m, n, dim) + np.random.randn(m, 1, dim)
    y = np.argmax(np.linalg.norm(x, axis=2, ord=2), axis=1)
    return x, y


class MaxValueExtractor(nn.Module):

    def __init__(self, dim=16):
        super().__init__()
        self.projection = nn.Linear(2, dim, bias=True)
        self.encoder_blocks = nn.Sequential(
            EncoderBlock(dim, 4 * dim, num_heads=1),
            EncoderBlock(dim, 4 * dim, num_heads=1),
        )
        self.pointer_attn = PointerAttention(dim, dim)
        self.query = nn.Parameter(torch.empty(1, dim))
        self.ln = RMSNorm(dim)
        nn.init.xavier_normal_(self.projection.weight)
        nn.init.xavier_normal_(self.query)

    def forward(self, x):
        bsz, tsz, _ = x.shape
        x = self.projection(x)
        q = self.query.repeat(bsz, 1, 1)
        x = self.encoder_blocks(x)
        x = self.ln(x)
        log_lik = self.pointer_attn(q, x).squeeze(dim=1)
        return log_lik


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--iterations", type=int, default=300)
    argparser.add_argument("--device", type=str, default="cpu")
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model = MaxValueExtractor(16).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iterations)

    x_tr, y_tr = gen_data(500, 50)
    x_tr = torch.tensor(x_tr, dtype=torch.float, device=args.device)
    y_tr = torch.tensor(y_tr, dtype=torch.float, device=args.device)

    for i in range(args.iterations):

        optimizer.zero_grad()
        y_hat = model.forward(x_tr)
        loss = -torch.take_along_dim(y_hat, y_tr.unsqueeze(-1).long(), dim=-1)
        loss.mean().backward()
        optimizer.step()
        scheduler.step()
        if i % 1 == 0:
            logger.info(f"Iter: {i}\t"
                        f"Loss: {loss.mean().data:.2f}\t")


    x_te, y_te = gen_data(100, 50)
    x_te = torch.tensor(x_te, dtype=torch.float32, device=args.device)
    y_te = torch.tensor(y_te, dtype=torch.int64, device=args.device)
    with torch.no_grad():
        y_hat = model.forward(x_te)
        y_hat_argmax = torch.argmax(y_hat, dim=-1)

    x_te = x_te.data.cpu().numpy()
    y_te = y_te.data.cpu().numpy()
    y_hat_argmax = y_hat_argmax.data.cpu().numpy()

    NUM_SAMPLES_TO_VIS = 6

    plt.figure(figsize=(13, 8))
    for i in range(NUM_SAMPLES_TO_VIS):
        plt.subplot(2, 3, i + 1)
        plt.scatter(x_te[i, :, 0], x_te[i, :, 1], marker="o", alpha=0.5, color="grey")
        plt.scatter(x_te[i, y_te[i].item(), 0], x_te[i, y_te[i].item(), 1], marker="o",
                    alpha=1.0, color="blue")
        plt.scatter(x_te[i, y_hat_argmax[i], 0], x_te[i, y_hat_argmax[i], 1], marker="o",
                    alpha=1.0, color="green" if y_hat_argmax[i] == y_te[i] else "red")
        plt.axvline(0, ls="--", color="black", alpha=0.5)
        plt.axhline(0, ls="--", color="black", alpha=0.5)
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)

    plt.tight_layout()
    plt.savefig("./examples/max_value.png")
    plt.show()

    logger.info(f"Accuracy: {(y_hat_argmax == y_te).mean()}")

