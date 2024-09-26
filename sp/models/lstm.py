import torch
import torch.nn as nn
import lightning as L


class LSTMBasedStockPred(L.LightningModule):

    def __init__(self, context=12, foresight=6):
        super().__init__()
        self.lstm = nn.LSTM(context, context)
        self.decoder = nn.Linear(context, foresight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, _ = self.lstm(x)
        y_hat = self.decoder(x1)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('valid_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
