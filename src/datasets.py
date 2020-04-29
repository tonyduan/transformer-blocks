from torchtext.experimental import datasets


def get_dataset(name, split):
    if name == "wikitext2":
        return datasets.WikiText2("./data/wikitext-2", f"wiki.{split}.tokens")
    if name == "imdb":
        return datasets.IMDB.iters(batch_size=32, root="./data/imdb", device="cpu")
