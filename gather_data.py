from datetime import datetime, timedelta
import sqlite3
import time

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame


LAST_FETCH = datetime.now() - timedelta(days=1)
FSTR = '%Y-%m-%d %H:%M:%S'


def get_data(symb) -> pd.DataFrame:
    '''Get hourly bars for $symb for last 24 hours.
    '''
    # No keys required for crypto data
    client = CryptoHistoricalDataClient()
    
    now = datetime.now()
    assert (now - LAST_FETCH) > timedelta(hours=6), 'Must be at least 6 hours between fetches.'
    print('Fetching data at', now.strftime(FSTR))
    
    # Creating request object
    request_params = CryptoBarsRequest(
      symbol_or_symbols=[symb],
      timeframe=TimeFrame.Hour,
      start=LAST_FETCH.strftime(FSTR),
      end=now.strftime(FSTR),
    )
    
    LAST_FETCH = now
    
    btc_bars = client.get_crypto_bars(request_params)
    
    return btc_bars.df.reset_index()


def append_to_db(df: pd.DataFrame, db_path: str, symbol: str):
    '''Append data to table indicated by symbol name.'''
    conn = sqlite3.connect('./data.db')
    df.to_sql(symbol, conn, if_exists='append')
    print(f'Appended data to DB in table {symbol}')


if __name__ == '__main__':
    symbol = 'BTC/USD'
    while True:
        df = get_data(symbol)
        append_to_db(df, './data.db', symbol)
        print('Waiting 6 hours...)
        time.sleep(6*3600)
