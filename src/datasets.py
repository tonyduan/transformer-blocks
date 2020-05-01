import math
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.experimental import datasets



def get_dataset(name, split, tokenizer="basic_english"):

    tokenizer = get_tokenizer(tokenizer)

    if name == "wikitext2":
        train, test, val = datasets.WikiText2(root="./data/", tokenizer=tokenizer)
        if split == "train":
            return train
        if split == "test":
            return test

    if name == "imdb":
        train, test = datasets.IMDB(root="./data/", tokenizer=tokenizer, data_select=split)
        if split == "train":
            return train
        if split == "test":
            return test


class LanguageModelDatasetWrapper(Dataset):
    """
    Wrapper around a language model dataset yielding x of `seq_length` and y the following token.

    Notes
    -----
    The length of this wrapped dataset will be length of original dataset minus sequence length.
    """
    def __init__(self, dataset, seq_length):
        self.dataset = dataset
        self.vocab = self.dataset.vocab
        self.seq_length = seq_length

    def __len__(self):
        return len(self.dataset) - self.seq_length

    def __getitem__(self, idx):
        return self.dataset[idx:idx + self.seq_length], self.dataset[idx + self.seq_length]

