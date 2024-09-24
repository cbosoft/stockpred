from typing import List
from argparse import ArgumentParser
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

from .data import StockData
from .models import MODELS


class DeploymentCallback(Callback):

    def on_fit_end(self, trainer, model):
        path = Path(trainer.log_dir) / 'final_model.pt'
        ts = model.to_torchscript()
        torch.jit.save(ts, str(path))


class PlotCallback(Callback):

    def on_fit_end(self, trainer, model):
        log_dir = Path(trainer.log_dir)
        metrics_path = log_dir / 'metrics.csv'
        df = pd.read_csv(metrics_path)

        steps = np.array(df['step'])
        train_loss = np.array(df['train_loss'])
        valid_loss = np.array(df['valid_loss'])
        train_steps = steps[np.isfinite(train_loss)]
        train_loss = train_loss[np.isfinite(train_loss)]
        valid_steps = steps[np.isfinite(valid_loss)]
        valid_loss = valid_loss[np.isfinite(valid_loss)]

        plt.figure()
        plt.plot(train_steps, train_loss, label='training')
        plt.plot(valid_steps, valid_loss, label='validation')
        plt.xlabel('Step [#]')
        plt.ylabel('Loss [-]')
        plt.legend()
        plt.tight_layout()
        plt.savefig(log_dir / 'loss_plot.png')



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

    pivot = int(len(data) * 0.8)
    train_data, valid_data = torch.utils.data.random_split(data, [pivot, len(data) - pivot])
    train_dl = DataLoader(train_data, batch_size=8, shuffle=True, drop_last=True)
    valid_dl = DataLoader(valid_data, batch_size=8, shuffle=True, drop_last=True)
    for model_name in args.models:
        model = MODELS[model_name](context=args.context, foresight=args.foresight)
        trainer = L.Trainer(logger=CSVLogger('.', name='runs'), callbacks=[DeploymentCallback(), ModelCheckpoint(monitor='valid_loss', every_n_epochs=100), PlotCallback()], max_epochs=10)
        trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
    return trainer


if __name__ == '__main__':
    trainer = main()
