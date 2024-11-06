import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod

"""
class Strategy: (abstract class?)
- given hlo or other information
- build a strategy with some hyperparameters
- calculates returns
- given class BackTest
- optimization of hyperparameters by maximizing one of the indicator from BackTest
"""


class Strategy(ABC):
    def __init__(self, _data: pd.DataFrame):
        self.price: str = 'Adj Close'
        self.df: pd.DataFrame = _data
        self.results_df: pd.DataFrame = pd.DataFrame()
        self.cash: float = 10_000
        self.start_date = None

        self.run(self.df)

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def signals(self) -> list:
        """
        calculate signals
        :return:
        """
        pass

    @abstractmethod
    def feed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    # ---- not in use right now ----
    def feed_cash(self, cash: float | int) -> None:
        if cash < 1000:
            raise ValueError('Cash must be greater than 1000')
        self.cash = cash

    def check_data(self, col_names: list) -> None:
        for col_name in col_names:
            if col_name not in self.df.columns:
                raise ValueError(f'{col_name} not in self.df')

    # ---- in use ----
    def run(self, _data: pd.DataFrame) -> pd.DataFrame:
        self.df = self.feed_data(_data)

        self.df['Signal'] = self.signals()

        self.df['Position'] = self.get_position()
        self.start_date = self.df['Position'][self.df['Position'] == 1].first_valid_index()
        print(self.start_date)
        self.df = self.df[self.df.index >= self.start_date]

        self.results_df['Price'] = self.get_price()  # asset price
        self.results_df['Value'] = self.get_strategy_value()  # strategy value according to the price

        self.results_df['Buy & Hold Return'] = self.get_return()
        self.results_df['Return'] = self.get_strategy_return()  # strategy return

        self.results_df['Cumulative Return'] = self.get_cumulative_return()
        self.results_df['Strategy Cumulative Return'] = self.get_strategy_cumulative_return()

        self.plot()

        return self.results_df.copy()

    # ---- connect wth backtest ----
    def pass_to_backtest(self) -> pd.DataFrame:
        return self.results_df[['Value', 'Return', 'Price']]

    def get_price(self) -> pd.Series:
        """
        asset price
        assumes df contains 'Adj Close' or 'Close'
        :return: The asset price series
        """
        return self.df[self.price]

    def get_return(self) -> pd.Series:
        """
        buy & hold return
        :return:
        """
        return self.get_price().pct_change()

    def get_strategy_value(self) -> pd.Series:
        """
        buy and hold value according to the asset value
        :return:
        """
        temp_df = pd.DataFrame()
        temp_df['Position'] = self.get_position()
        temp_df['Value'] = np.where(
            temp_df['Position'] == 0, self.get_price().iloc[0],
            self.get_strategy_cumulative_return() * self.get_price().iloc[0]
        )

        return temp_df['Value']

    def get_signal(self) -> pd.Series:
        """
        when to buy or sell or hold at the time
        assumes df contains 'Signal'
        :return:
        """
        return self.df['Signal']

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

    def get_strategy_cumulative_return(self) -> pd.Series:
        return np.cumprod(1 + self.get_strategy_return())

    # ---- strategy optimisation ----
    @abstractmethod
    def optimise(self) -> None:
        pass

    # ---- Visualisation ----
    def plot(self) -> None:
        """
        the main plot functions, do not overwrite
        :return:
        """
        self.plot_graph()
        self.plot_show()
        #
        # self.plot_cumulative_return()

    def plot_graph(self) -> None:
        """
        overwrite this function if needed
        :return:
        """
        # Plot the price data with buy and sell signals
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.get_price(), label='Price')

        # Plotting buy signals
        ax.plot(
            self.df[self.df['Signal'].diff() >= 1].index, self.df[self.df['Signal'].diff() >= 1][self.price],
            '^', color='g', label='Buy Signal'
        )  # , markersize=10

        # Plotting sell signals
        ax.plot(
            self.df[self.df['Signal'].diff() <= -1].index, self.df[self.df['Signal'].diff() <= -1][self.price],
            'v', color='r', label='Sell Signal'
        )

    def plot_cumulative_return(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.results_df['Cumulative Return'], label='Buy & Hold Cumulative Return')
        ax.plot(self.results_df['Strategy Cumulative Return'], label='Strategy Cumulative Return')

        self.plot_show()

    def plot_show(self) -> None:
        plt.title(f'Strategy: {self}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

class MovingAverageCrossOver(Strategy):
    def __str__(self):
        return 'moving_average_crossover'

    def __init__(self, _data: pd.DataFrame, fast: int = 10, slow: int = 50):
        super().__init__(_data)

        # -- format check --
        assert 0 < fast < slow

        self.fast: int = fast
        self.slow: int = slow

    def signals(self) -> list:
        # fill na with 0, if fast MA > slow MA, signal = 1, else -1
        signals = []
        be4_first_trend: bool = True

        for i in range(self.df.shape[0]):
            up_trend: bool = self.df['Slow'].iloc[i] < self.df['Fast'].iloc[i]
            prev_down: bool = self.df['Slow'].iloc[i - 1] >= self.df['Fast'].iloc[i - 1]

            if up_trend and prev_down:
                be4_first_trend = False

            if be4_first_trend:
                signals.append(0)
                continue

            signals.append(1 if up_trend else -1)

        return signals

    def feed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # -- format check --
        if 'Adj Close' in df.columns:
            self.price = 'Adj Close'
        elif 'Close' in df.columns:
            self.price = 'Close'
            print('Adj Close not found. Using Close price instead of Adj Close.')
        else:
            raise ValueError('Please ensure that the columns contain Adj Close or Close.')

        # -- feed data --
        df = df.copy()
        df['Fast'] = df[f'{self.price}'].rolling(self.fast).mean()
        df['Slow'] = df[f'{self.price}'].rolling(self.slow).mean()

        return df

    def optimise(self) -> None:
        best_fast: int = 1
        best_slow: int = 2
        # TODO: Finish optimisation
        print(f'Best params:\nFast: {best_fast} Slow: {best_slow}')

    def plot_graph(self) -> None:
        super().plot_graph()
        plt.plot(self.df['Fast'], color='y', label='Fast')
        plt.plot(self.df['Slow'], color='purple', label='Slow')


class EmaCrossover(MovingAverageCrossOver):
    def feed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Fast'] = df[f'{self.price}'].ewm(span=self.fast, adjust=False).mean()
        df['Slow'] = df[f'{self.price}'].ewm(span=self.slow, adjust=False).mean()

        return df

class MeanReversion(Strategy):
    def __init__(
            self,
            _data: pd.DataFrame,
            _ma_length: int = 50,
            _days: int = 200,
            _up: float = 1,
            _down: float = 1
    ):
        super().__init__(_data)

        assert _days > 0
        assert _ma_length > 0
        assert _up > 0
        assert _down > 0

        self.days: int = _days
        self.ma_length: int = _ma_length
        self.up: float = _up
        self.down: float = _down

    def feed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df['ma'] = self.get_price().rolling(self.ma_length).mean()
        df['risk'] = self.get_price().rolling(self.days).std()
        df['up_bound'] = self.get_price() + self.up * df['risk']
        df['down_bound'] = self.get_price() - self.down * df['risk']

        return df

    def signals(self) -> list:
        prices = self.df.get_price()
        n = self.df.shape[0]
        signals = [None] * n
        for i in range(n):
            if i == 0:
                signals[i] = 0

            elif self.df.low_bound.iloc[i] < prices.iloc[i] < self.df.up_bound.iloc[i]:
                signals[i] = signals[i - 1]

            elif prices.iloc[i] <= self.df.low_bound.iloc[i]:
                if signals[i - 1] == 0:
                    signals[i] = 1
                elif signals[i - 1] == -1:
                    signals[i] = 0
                else:
                    signals[i] = signals[i - 1]

            else:
                if signals[i - 1] == 0:
                    signals[i] = -1
                elif signals[i - 1] == 1:
                    signals[i] = 0
                else:
                    signals[i] = signals[i - 1]

        return signals

    def optimise(self) -> None:
        pass

    def __str__(self):
        return 'mean_reversion'

