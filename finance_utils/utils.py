import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance as yf


def get_price(tickers, start_date, end_date):  # get adj close
    df = yf.download(tickers, start=start_date, end=end_date)
    # if no. of tickers > 1, then df have multi-index columns
    if isinstance(tickers, list):
        price = df['Adj Close']
        price.columns = tickers
        df = price

    return df


def mean(x):
    return x.mean()


def var(x):
    return x.var()


def std(x):
    return x.std()


def skew(x):
    return stats.skew(x)


def kert(x):
    return stats.kurtosis(x)


