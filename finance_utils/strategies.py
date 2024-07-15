import numpy as np
import pandas as pd
from backtest import Backtest


# TODO: Need to make this an abstract class
class Strategies:
    def __init__(self):
        self.benchmark: str = 'Adj Close'
        self.df: pd.DataFrame = pd.DataFrame()
        self.cash: float = 10_000

    def feed_data(self, df: pd.DataFrame) -> None:
        self.df = df

    def feed_cash(self, cash: float | int) -> None:
        if cash < 1000:
            raise ValueError('Cash must be greater than 1000')
        self.cash = cash

    def set_benchmark(self, benchmark: str) -> None:
        self.benchmark = benchmark

    def run(self) -> Backtest:
        results_df = pd.DataFrame()
        results_df['Value'] = self.get_price()
        results_df['Return'] = self.get_return()
        # TODO: Need to find a way to let users customise the start, end date, risk-free rate & benchmark
        return Backtest(results_df, results_df.index.iloc[0], results_df.index.iloc[-1])

    def get_price(self) -> pd.Series:
        pass

    def get_return(self) -> pd.Series:
        pass

    def get_signal(self) -> pd.Series:
        pass

    def get_position(self) -> pd.Series:
        return self.get_signal().shift(1).fillna(0)

    def get_strategy_return(self) -> pd.Series:
        return self.get_return() * self.get_position()

    def get_cumulative_return(self) -> pd.Series:
        return (1 + self.get_strategy_return()).cumprod()


def fast_slow(df_prev: pd.DataFrame, fast: int, slow: int, ticker_name: str = None) -> pd.DataFrame:
    # _df contains the Close of a stock
    if fast < 0: raise ValueError('Fast must be greater than 0')
    if slow < 0: raise ValueError('Slow must be greater than 0')
    if fast > slow:
        temp = fast
        fast = slow
        slow = temp
    if ticker_name is None:
        ticker_name = 'Close'
    _df = pd.DataFrame(df_prev[ticker_name])
    _df.dropna(inplace=True)

    _df[f'MA{fast}'] = _df[ticker_name].rolling(fast).mean()
    _df[f'MA{slow}'] = _df[ticker_name].rolling(slow).mean()

    _df['Signal'] = np.where(
        _df[f'MA{slow}'].isna(),
        0,
        np.where(_df[f'MA{slow}'] < _df[f'MA{fast}'], 1, -1)
    )  # fill na with 0, if fast MA > slow MA, signal = 1, else -1
    _df['Position'] = _df['Signal'].shift(1)
    _df['Strategy Return'] = _df[ticker_name].pct_change() * _df['Position']
    _df['Cumulative Return'] = (1 + _df['Strategy Return']).cumprod()

    # return a dataframe with columns: Price, Strategy Return, Cumulative Return
    # it also contains other stuff, but less important
    return _df


# need to redo the ema functions
def ema_fast_slow(df_prev: pd.DataFrame, fast: int, slow: int, ticker_name: str = None) -> pd.DataFrame:
    if fast < 0: raise ValueError('Fast must be greater than 0')
    if slow < 0: raise ValueError('Slow must be greater than 0')
    if fast > slow:
        temp = fast
        fast = slow
        slow = temp
    if ticker_name is None:
        ticker_name = 'Close'
    _df = pd.DataFrame(df_prev[ticker_name])
    _df.dropna(inplace=True)

    _df[f'EMA{fast}'] = _df[ticker_name].ewm(span=fast, adjust=False).mean()
    _df[f'EMA{slow}'] = _df[ticker_name].ewm(span=slow, adjust=False).mean()

    _df['Signal'] = np.where(
        _df[f'EMA{slow}'].isna(),
        0,
        np.where(_df[f'EMA{slow}'] < _df[f'EMA{fast}'], 1, -1)
    )  # fill na with 0, if fast MA > slow MA, signal = 1, else -1
    _df['Position'] = _df['Signal'].shift(1)
    _df['Strategy Return'] = _df[ticker_name].pct_change() * _df['Position']
    _df['Cumulative Return'] = np.cumprod(1 + _df['Strategy Return']) - 1

    return _df


def buy_and_hold(df_prev: pd.DataFrame, ticker_name: str = None) -> pd.DataFrame:
    if ticker_name is None:
        ticker_name = 'Close'
    _df = pd.DataFrame(df_prev[ticker_name])
    _df.dropna(inplace=True)

    _df['Strategy Return'] = _df[ticker_name].pct_change()
    _df['Cumulative Return'] = (1 + _df['Strategy Return']).cumprod()

    return _df
