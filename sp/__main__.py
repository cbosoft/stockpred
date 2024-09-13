from typing import List
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import CSVLogger

from .data import StockData
from .models import MODELS


def list_of_str(s: str) -> List[str]:
    return s.split(',')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--context', '-c', type=int, default=12)
    parser.add_argument('--foresight', '-f', type=int, default=6)
    parser.add_argument('--models', '-m', type=list_of_str, default='lstm')
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_float32_matmul_precision('high')
    data = StockData(context=args.context, foresight=args.foresight)
    dl = DataLoader(data)
    trainer = L.Trainer(logger=CSVLogger('.', name='runs'))
    for model_name in args.models:
        model = MODELS[model_name](context=args.context, foresight=args.foresight)
        trainer.fit(model=model, train_dataloaders=dl)


if __name__ == '__main__':
    main()
