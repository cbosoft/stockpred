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
        plt.close()


class PredPlotter(Callback):

    def __init__(self, data, context, foresight):
        super().__init__()
        self.data = data
        self.context = context
        self.foresight = foresight

    def on_fit_end(self, trainer, model):
        model.eval()

        n = len(self.data)
        data = np.array(self.data) # create a copy

        for i in range(0, n - self.context, self.foresight):
            x = data[i:i+self.context]
            ref = x[-1]
            x = (x - ref) / ref
            y_hat_i = model(torch.tensor(x).float().unsqueeze(0)).cpu().detach().numpy().squeeze(0)
            lb = i + self.context
            ub = min(n, i + self.context + self.foresight)
            data[lb:ub] = y_hat_i[:ub-lb] * ref + ref

        c = self.context - 1
        i = np.arange(n)
        left_i = i[:c]
        right_i = i[c:]

        plt.figure()
        plt.plot(left_i, self.data[:c], 'C0-')
        plt.plot(right_i, self.data[c:], 'C0--')
        plt.plot(right_i, data[c:], 'C1-')
        log_dir = Path(trainer.log_dir)
        plt.savefig(log_dir / 'pred_plot.png')
        plt.save()


def list_of_str(s: str) -> List[str]:
    return s.split(',')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--context', '-c', type=int, default=12)
    parser.add_argument('--foresight', '-f', type=int, default=6)
    parser.add_argument('--models', '-m', type=list_of_str, default='lstm')
    parser.add_argument('--epochs', '-e', type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_float32_matmul_precision('high')
    train_data, valid_data, test_data = StockData.load(context=args.context, foresight=args.foresight)

    train_dl = DataLoader(train_data, batch_size=8, shuffle=True)
    valid_dl = DataLoader(valid_data, batch_size=8)
    test_dl = DataLoader(test_data)

    for model_name in args.models:
        model = MODELS[model_name](context=args.context, foresight=args.foresight)
        trainer = L.Trainer(logger=CSVLogger('.', name='runs'), callbacks=[DeploymentCallback(), ModelCheckpoint(monitor='valid_loss', every_n_epochs=100), PlotCallback(), PredPlotter(test_data.data, args.context, args.foresight)], max_epochs=args.epochs)
        trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
        trainer.test(model=model, dataloaders=test_dl)
    return trainer


if __name__ == '__main__':
    trainer = main()
