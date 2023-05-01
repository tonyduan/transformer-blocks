import logging
import pathlib
import pickle
import os
import numpy as np
import torch
import torch.optim as optim
from argparse import ArgumentParser
from torchnet import meter
from torch.utils.data import DataLoader
from src.models import BERTLanguageModel
from src.datasets import LanguageModelDatasetWrapper, get_dataset


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--device", default="cuda", type=str)
    argparser.add_argument("--lr", default=1e-4, type=float)
    argparser.add_argument("--batch-size", default=32, type=int)
    argparser.add_argument("--seq-length", default=32, type=int)
    argparser.add_argument("--num-layers", default=4, type=int)
    argparser.add_argument("--dim", default=768, type=int)
    argparser.add_argument("--hidden-dim", default=768, type=int)
    argparser.add_argument("--num-workers", default=min(os.cpu_count(), 8), type=int)
    argparser.add_argument("--num-epochs", default=20, type=int)
    argparser.add_argument("--print-every", default=20, type=int)
    argparser.add_argument("--save-every", default=5, type=int)
    argparser.add_argument("--experiment-name", default="wikitext2", type=str)
    argparser.add_argument("--dataset", default="wikitext2", type=str)
    argparser.add_argument('--output-dir', type=str, default=os.getenv("PT_OUTPUT_DIR"))
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    dataset = get_dataset(args.dataset, "train")
    dataset = LanguageModelDatasetWrapper(dataset, args.seq_length)
    train_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size,
                              num_workers=args.num_workers)

    logger.info(f"Vocabulary size: {len(dataset.vocab.itos)}")

    model = BERTLanguageModel(device=args.device,
                              vocab_size=len(dataset.vocab.itos),
                              num_layers=args.num_layers,
                              dim=args.dim,
                              hidden_dim=args.hidden_dim,
                              num_heads=8,
                              dropout_prob=0.1,
                              max_length=args.seq_length)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4,
                          nesterov=True)
    annealer = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)

    loss_meter = meter.AverageValueMeter()
    time_meter = meter.TimeMeter(unit=False)

    train_losses = []

    for epoch in range(args.num_epochs):

        for i, (x, y) in enumerate(train_loader):

            x, y = x.to(args.device), y.to(args.device)
            loss = model.loss(x, y).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.cpu().data.numpy(), n=1)

            if i % args.print_every == 0:
                logger.info(f"Epoch: {epoch}\t"
                            f"Itr: {i} / {len(train_loader)}\t"
                            f"Loss: {loss_meter.value()[0]:.2f}\t"
                            f"Mins: {(time_meter.value() / 60):.2f}\t"
                            f"Experiment: {args.experiment_name}")
                train_losses.append(loss_meter.value()[0])
                loss_meter.reset()

        if (epoch + 1) % args.save_every == 0:
            save_path = f"{args.output_dir}/{args.experiment_name}/{epoch + 1}/"
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"{save_path}/model_ckpt.torch")

        annealer.step()

    pathlib.Path(f"{args.output_dir}/{args.experiment_name}").mkdir(parents=True, exist_ok=True)
    save_path = f"{args.output_dir}/{args.experiment_name}/model_ckpt.torch"
    torch.save(model.state_dict(), save_path)
    args_path = f"{args.output_dir}/{args.experiment_name}/args.pkl"
    pickle.dump(args, open(args_path, "wb"))
    save_path = f"{args.output_dir}/{args.experiment_name}/losses_train.npy"
    np.save(save_path, np.array(train_losses))

