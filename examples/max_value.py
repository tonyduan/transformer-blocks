import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.blocks import ISAB, PMA
from argparse import ArgumentParser
from matplotlib import pyplot as plt


def gen_data(m, n, dim=2):
    """
    m batches of n samples each ~ N, and of dimension 2.
    """
    x = np.random.randn(m, n, dim) + 5 * np.random.randn(m, 1, dim)
    idxs = np.argmax(np.linalg.norm(x, axis=2, ord=2), axis=1)
    y = np.array([x[i] for x,i in zip(x,idxs)])
    return x, y[:,np.newaxis,:]

if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--iterations", type=int, default=300)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    x_tr, y_tr = gen_data(500, 50)

    x_tr = torch.tensor(x_tr, dtype=torch.float)
    y_tr = torch.tensor(y_tr, dtype=torch.float)

    model = nn.Sequential(ISAB(dim=2, num_heads=1, num_inds=1),
                          PMA(dim=2, num_heads=1))
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    for i in range(args.iterations):
        optimizer.zero_grad()
        y_hat = model.forward(x_tr)
        loss = torch.norm(y_hat - y_tr, dim=2)
        loss.mean().backward()
        optimizer.step()
        if i % 1 == 0:
            logger.info(f"Iter: {i}\t"
                        f"Loss: {loss.mean().data:.2f}\t")

    x_te, y_te = gen_data(2, 50)
    x_te = torch.tensor(x_te, dtype=torch.float)
    y_te = torch.tensor(y_te, dtype=torch.float)
    y_hat = model.forward(x_te)

    plt.figure(figsize=(8, 3))
    plt.scatter(x_te.data[:,:,0], x_te.data[:,:,1], color="black",
                label="Train", marker="x", alpha=0.5)
    plt.scatter(y_hat.data[:,:,0], y_hat.data[:,:,1], color="red",
                label="Pred", marker="x", alpha=0.5)
    plt.scatter(y_te.data[:,:,0], y_te.data[:,:,1], color="blue",
                label="True", marker="x", alpha=0.5)
    plt.axvline(0, ls="--", color="black", alpha=0.5)
    plt.axhline(0, ls="--", color="black", alpha=0.5)
    plt.legend()

    plt.savefig("./examples/max_value.png")
    plt.show()

