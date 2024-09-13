import torch
import torch.nn as nn
import lightning as L


class LSTMBasedStockPred(L.LightningModule):

    def __init__(self, context=12, foresight=6):
        super().__init__()
        self.lstm = nn.LSTM(context, context)
        self.decoder = nn.Linear(context, foresight)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x1, _ = self.lstm(x)
        y_hat = self.decoder(x1)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
