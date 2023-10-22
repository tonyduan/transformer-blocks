from argparse import ArgumentParser
import logging

import torch
import torch.nn as nn

from src.blocks import EncoderBlock


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--bsz", type=int, default=4)
    argparser.add_argument("--tsz", type=int, default=16)
    argparser.add_argument("--dim", type=int, default=384)
    argparser.add_argument("--num-layers", type=int, default=120)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    blocks = nn.ModuleList([
        EncoderBlock(args.dim, 4 * args.dim)
        for _ in range(args.num_layers)
    ])

    x = torch.randn((args.bsz, args.tsz, args.dim)) * args.dim ** -0.5

    print(x.min(), x.max(), x.norm(dim=-1).mean())

    with torch.no_grad():
        for block in blocks:
            x = block(x)

    print(x.min(), x.max(), x.norm(dim=-1).mean())
