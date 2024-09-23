import sqlite3

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class StockData(Dataset):

    def __init__(self, context=12, foresight=6):
        conn = sqlite3.connect('data.db')
        df = pd.read_sql('SELECT * FROM "BTC/USD";', conn)
        close = df['close'].values.astype('float32')
        delta_close = np.diff(close)
        self.close_price = torch.tensor(delta_close)
        self.context = context
        self.foresight = foresight

    def __len__(self) -> int:
        return len(self.close_price) - self.context - self.foresight

    def __getitem__(self, i: int):
        inp = self.close_price[ i               : i + self.context                  ]
        tgt = self.close_price[ i + self.context: i + self.context + self.foresight ]
        return inp, tgt

