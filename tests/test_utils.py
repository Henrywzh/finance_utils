from datetime import datetime
from finance_utils.indicators import *
from finance_utils.display import *
import yfinance as yf

tickers = ['NVDA', 'AAPL', 'MSFT', 'GOOG', 'TSLA']
df = yf.download(tickers, start=datetime(2023, 1, 1), end=datetime(2024, 7, 1))

df_price = df.pivot(index='Date', columns='Tickers', values='Close')



