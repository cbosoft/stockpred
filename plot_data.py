import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


conn = sqlite3.connect('data.db')
df = pd.read_sql('SELECT * FROM "BTC/USD";', conn)
df['timestamp'] = df['timestamp'].map(lambda s: datetime.strptime(s.split('+')[0], '%Y-%m-%d %H:%M:%S'))

df['mean'] = (df['open'] + df['close'])*0.5
plt.plot(df['timestamp'], df['mean'])
plt.fill_between(df['timestamp'], df['low'], df['high'], alpha=0.5)
plt.xlabel('Date')
plt.ylabel('BTC ($)')
plt.savefig('price.png', dpi=300)
plt.close()

# correlation
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df['mean'], lags=100)
#plt.show()
plt.close()

import seaborn as sns
sns.pairplot(df)
plt.show()
