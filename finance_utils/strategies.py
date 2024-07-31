import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from backtest import Backtest


# TODO: Need to make this an abstract class
class Strategies:
    def __init__(self):
        self.benchmark: str = 'Adj Close'
        self.df: pd.DataFrame = pd.DataFrame()
        self.results_df: pd.DataFrame = pd.DataFrame()
        self.cash: float = 10_000

    def feed_data(self, df: pd.DataFrame) -> None:
        self.df = df

    def feed_cash(self, cash: float | int) -> None:
        if cash < 1000:
            raise ValueError('Cash must be greater than 1000')
        self.cash = cash

    def set_benchmark(self, benchmark: str) -> None:
        self.benchmark = benchmark

    def run(self) -> pd.DataFrame:
        self.results_df['Price'] = self.get_price()
        self.results_df['Value'] = self.get_value()
        self.results_df['Return'] = self.get_return()

        self.plot()

        # TODO: Need to find a way to let users customise the start, end date, risk-free rate & benchmark
        return self.results_df.copy()

    def get_price(self) -> pd.Series:
        """
        asset price
        :return:
        """
        pass

    def get_return(self) -> pd.Series:
        """
        buy & hold return
        :return:
        """
        return self.get_price().pct_change()

    def buy_and_hold_value(self) -> pd.Series:
        # TODO: lets say buy the stock twice at different time, need to count in the second transaction to the benchmark
        pass

    def get_value(self) -> pd.Series:
        # TODO: Actually I'm not sure, only use this function when you buy sell hold once
        """
        strategy value according to the asset value
        :return:
        """
        return self.get_cumulative_return() * self.get_price()

    def get_signal(self) -> pd.Series:
        """
        when to buy or sell or hold at the time
        :return:
        """
        pass

    def get_position(self) -> pd.Series:
        """
        whether long or short at the time
        :return:
        """
        return self.get_signal().shift(1).fillna(0)

    def get_strategy_return(self) -> pd.Series:
        return self.get_return() * self.get_position()

    def get_cumulative_return(self) -> pd.Series:
        return np.cumprod(1 + self.get_return())

    # ---- Visualisation ----
    def plot(self) -> None:
        # Plot the price data with buy and sell signals
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.get_price(), label='Price')

        signals = self.get_signal()

        # Plotting buy signals
        ax.plot(
            signals.loc[signals == 1.0].index, signals[signals == 1.0],
            '^', markersize=10, color='g', label='Buy Signal'
        )

        # Plotting sell signals
        ax.plot(
            signals.loc[signals == -1.0].index, signals[signals == -1.0],
            'v', markersize=10, color='r', label='Sell Signal'
        )

        plt.title('Strategy')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()


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
