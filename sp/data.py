import sqlite3
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class StockData(Dataset):

    def __init__(self, data, context, foresight):
        self.data = data
        self.context = context
        self.foresight = foresight

    @classmethod
    def load(cls, context=12, foresight=6, train_frac=0.6, valid_frac=0.2) -> List["StockData"]:
        conn = sqlite3.connect('data.db')
        df = pd.read_sql('SELECT * FROM "BTC/USD";', conn)
        close = np.array(df['close']).astype('float32')
        n = len(close)
        n_train = int(n*train_frac)
        n_valid = int(n*valid_frac)
        train = torch.tensor(close[:n_train])
        valid = torch.tensor(close[n_train:n_train+n_valid])
        test = torch.tensor(close[n_train+n_valid:])
        assert min([len(train), len(valid), len(test)]) > (context + foresight), f'Not enough data for selected context and foresight values. (got {n} datapoints, cannot exceed that value).'
        return [
            cls(data, context, foresight)
            for data in [train, valid, test]
        ]

    def __len__(self) -> int:
        return len(self.data) - self.context - self.foresight

    def __getitem__(self, i: int):
        inp = self.data[ i               : i + self.context                  ]
        tgt = self.data[ i + self.context: i + self.context + self.foresight ]
        ref = inp[-1]
        inp = (inp - ref) / ref
        tgt = (tgt - ref) / ref
        return inp, tgt

