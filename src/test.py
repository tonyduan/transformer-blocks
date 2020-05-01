import numpy as np
import logging
import pathlib
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from torchnet import meter
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from src.models import BERTLanguageModel
from src.datasets import *


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--device", default="cuda", type=str)
    argparser.add_argument("--batch-size", default=32, type=int)
    argparser.add_argument("--seq-length", default=32, type=int)
    argparser.add_argument("--num-layers", default=4, type=int)
    argparser.add_argument("--dim", default=768, type=int)
    argparser.add_argument("--hidden-dim", default=768, type=int)
    argparser.add_argument("--num-workers", default=min(os.cpu_count(), 8), type=int)
    argparser.add_argument("--dataset-skip", default=1, type=int)
    argparser.add_argument("--experiment-name", default="wikitext2", type=str)
    argparser.add_argument("--dataset", default="wikitext2", type=str)
    argparser.add_argument("--output-dir", type=str, default=os.getenv("PT_OUTPUT_DIR"))
    argparser.add_argument("--save-path", type=str, default=None)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO, filename="out/examples.txt")
    logger = logging.getLogger(__name__)

    dataset = get_dataset(args.dataset, "test")
    vocabulary = dataset.vocab
    dataset = LanguageModelDatasetWrapper(dataset, args.seq_length)
    dataset = Subset(dataset, list(range(0, len(dataset), args.dataset_skip)))
    test_loader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, 
                             num_workers=args.num_workers)

    if not args.save_path:
        save_path = f"{args.output_dir}/{args.experiment_name}/model_ckpt.torch"
    else:
        save_path = args.save_path

    model = BERTLanguageModel(device=args.device,
                              vocab_size=len(vocabulary.itos),
                              num_layers=args.num_layers,
                              dim=args.dim,
                              hidden_dim=args.hidden_dim,
                              num_heads=8,
                              dropout_prob=0.1,
                              max_length=args.seq_length)

    saved_dict = torch.load(save_path)
    model.load_state_dict(saved_dict)
    model.eval()

    results = {
        "preds_nll": np.zeros(len(dataset)),
    }

    for i, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):

        x, y = x.to(args.device), y.to(args.device)
        logits = model.forward(x)
        loss = F.cross_entropy(logits, y, reduction="none")

        lower, upper = i * args.batch_size, (i + 1) * args.batch_size
        results["preds_nll"][lower:upper] = loss.data.cpu().numpy()

        # for now only log the first sequence in each mini-batch
        x, y = x.to("cpu"), y.to("cpu")
        y = vocabulary.itos[y[0]]
        x = " ".join([vocabulary.itos[t] for t in x[0]])
        y_hat = vocabulary.itos[logits[0].argmax().data.cpu()]
        logger.info(f"y: {y}\tyhat: {y_hat}\tx: {x}")

    print(f"NLL: {np.mean(results['preds_nll']):.2f}")

    save_path = f"{args.output_dir}/{args.experiment_name}"
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    for k, v in results.items():
        np.save(f"{save_path}/{k}.npy", v)

