from argparse import ArgumentParser

from matplotlib import pyplot as plt

from src.blocks import PositionalEmbedding


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--n", type=int, default=100)
    argparser.add_argument("--dim", type=int, default=20)
    args = argparser.parse_args()

    embed = PositionalEmbedding.make_positional_embedding(dim=args.dim, max_length=args.n).numpy()

    plt.figure(figsize=(8, 2))
    plt.imshow(embed.T)
    plt.xlabel("Tokens")
    plt.ylabel("Dimension")
    plt.tight_layout()
    plt.savefig("examples/vis_pos_emb.png")
    plt.show()
