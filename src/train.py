import torch
from src.datasets import *


if __name__ == "__main__":
    
    dataset = get_dataset("wikitext2", "train")
