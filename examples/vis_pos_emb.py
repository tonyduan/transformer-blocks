from argparse import ArgumentParser

from matplotlib import pyplot as plt
import numpy as np
import torch

from src.blocks import PositionalEncoding


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--n", type=int, default=100)
    argparser.add_argument("--dim", type=int, default=20)
    args = argparser.parse_args()

    encoding = PositionalEncoding(dim=args.dim, max_length=args.n)
    encoding = encoding.encoding.numpy()

    plt.figure(figsize=(8, 3))
    plt.imshow(encoding.T)
    plt.savefig("examples/vis_pos_emb.png")
    plt.xlabel("Tokens")
    plt.ylabel("Dimension")
    plt.show()
