import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from matplotlib import pyplot as plt

"""
class Portfolio:
- weights
- tickers name
- benchmark
- price & returns of all these assets
- value of your wealth
- founding time
- class BackTest -> to give some results related to the portfolio

class Strategy: (abstract class?)
- given hlo or other information
- build a strategy with some hyperparameters
- calculates returns
- given class BackTest
- optimization of hyperparameters by maximizing one of the indicator from BackTest
"""


class Portfolio:
    def __init__(
            self,
            data: pd.DataFrame,
            cash: int | float,
            start_date: str = None,
            end_date: str = None,
            benchmark: str = None,
            default_weights: list[float] = None,
    ):
        """
        :param data: daily stocks prices, columns: stock names
        :param cash: initial capital
        "param stocks: list of stocks in the portfolio
        :param default_weights: weights of each stock in the portfolio
        """
        # -- format check ---
        if cash < 10_000:
            raise ValueError("Please enter cash amount >= 10_000")

        if start_date and end_date and (start_date > end_date):
            raise ValueError("start_date should be less than end_date")

        if default_weights and len(data.columns) != len(default_weights):
            raise ValueError("number of stocks != length of default_weights")

        # -- initialisation --
        self.tickers = data.columns.tolist()
        self.prices = data
        self.shares = pd.DataFrame(index=self.prices.index, columns=self.tickers)
        self.cash = cash

        self.cashes: pd.DataFrame | list = [-1] * len(self.prices)
        self.portfolio_values: pd.DataFrame | list = [-1] * len(self.prices)

        # set benchmark
        self.benchmark = '^GSPC' if benchmark is None else benchmark

        # set date
        self.start_date = data.index[0] if start_date is None else pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(start_date) if end_date else data.index[-1]

        self.benchmark_prices = self._get_benchmark()
        self.weights = default_weights if default_weights else [1 / len(self.tickers) for t in self.tickers]

    # -- run --
    def run(self, period: int = 20) -> pd.DataFrame:
        self.rebalance_portfolio(period=period)

        return self.portfolio_values

    # -- set up, changes to portfolio settings --
    def reset(self) -> None:
        self.cashes = [-1] * len(self.prices)
        self.portfolio_values = [-1] * len(self.prices)

        self.benchmark_prices = self._get_benchmark()

    def set_start_date(self, start_date: str) -> None:
        _start_date = pd.to_datetime(start_date)
        if _start_date >= self.end_date:
            raise ValueError("start_date should be less than end_date")

        self.start_date = _start_date

    def set_end_date(self, end_date: str) -> None:
        _end_date = pd.to_datetime(end_date)
        if _end_date <= self.start_date:
            raise ValueError("end_date should be greater than start_date")

        self.end_date = _end_date

    def set_dates(self, start_date: str, end_date: str) -> None:
        _start_date = pd.to_datetime(start_date)
        _end_date = pd.to_datetime(end_date)

        if _start_date > _end_date:
            raise ValueError("start_date should be less than end_date")

        self.start_date = _start_date
        self.end_date = _end_date

    def add_stock(self, item: str | list[str] | pd.Series | pd.DataFrame) -> None:
        if isinstance(item, str):
            self.tickers += [item]
            self._download(item)
        elif isinstance(item, list):
            self.tickers += item
            self._download(item)
        elif isinstance(item, pd.Series):
            self.tickers += [item.name]
            self.prices = self.prices.join(item)
        elif isinstance(item, pd.DataFrame):
            self.tickers += list(item.columns)
            self.prices = self.prices.join(item)
        else:
            raise TypeError("item should be the following type: str, list, pd.Series, pd.DataFrame")

        self._reset_weights()
        # TODO:
        # need to check column format

    # -- quantitative analysis --
    def get_value(self, i: int | str) -> float:
        stocks_val = self.shares.iloc[i] * self.prices.iloc[i] if isinstance(i, int) \
            else self.shares.loc[i] * self.prices.loc[i]
        total_val = self.cash + stocks_val.sum()
        return total_val

    def get_portfolio_values(self) -> pd.DataFrame:
        return self.portfolio_values

    def get_portfolio_returns(self) -> pd.DataFrame:
        # pass the portfolio returns
        return self.portfolio_values.pct_change()

    def get_all_returns(self) -> pd.DataFrame:
        # return: daily returns of every stock in the portfolio
        return self.prices.pct_change()

    def get_portfolio_risk(self):
        # TODO:
        pass

    # -- visualisation --
    def plot(self):
        self.plot_cum_returns()
        self.plot_cum_values()
        self.plot_shares()
        self.plot_weights()

    def plot_cum_returns(self) -> None:
        (self.portfolio_values / self.portfolio_values.iloc[0]).plot(figsize=(10, 6))
        plt.title('Cumulative Returns')
        plt.show()

    def plot_cum_values(self) -> None:
        (self.shares * self.prices).plot(figsize=(10, 6))
        plt.title('Cumulative Values')
        plt.show()

    def plot_shares(self) -> None:
        self.shares.plot(figsize=(10, 6))
        plt.title('Shares')
        plt.show()

    def plot_weights(self) -> None:
        weights: pd.DataFrame = (self.prices * self.shares) / self.portfolio_values.values
        weights.plot(figsize=(10, 6))
        plt.title('Weights')
        plt.show()

    # -- portfolio management --

    def rebalance_portfolio(self, period: int = 20) -> None:
        # return: the daily value of the portfolio from start to end
        for i, d in enumerate(self.prices.index):
            if i == 0:
                self.portfolio_values[i] = self.cash
                for s, w in zip(self.tickers, self.weights):
                    self.shares.loc[d, s] = np.floor(self.portfolio_values[i] * np.array(w) / self.prices[s].iloc[i])

                self.cashes[i] = self._compute_cash(i)
                continue

            self.portfolio_values[i] = self._compute_values(i)

            if i % period == 0:
                self.shares.loc[d] = self._compute_shares(i)
            else:
                self.shares.loc[d] = self.shares.iloc[i - 1]

            self.cashes[i] = self._compute_cash(i)

        self.portfolio_values = pd.DataFrame(self.portfolio_values, columns=['Value'], index=self.prices.index)
        self.cashes = pd.DataFrame(data=self.cashes, columns=['Cash'], index=self.prices.index)

    def optimise_portfolio(self) -> list:
        # TODO:
        pass

    # -- private methods --
    def _download(self, ticker: str | list[str]) -> None:
        _df = yf.download(ticker, self.start_date, self.end_date)
        self.prices = self.prices.join(_df)
        # TODO: Debug

    def _get_benchmark(self) -> pd.Series:
        _df: pd.Series = yf.download(self.benchmark, self.start_date, self.end_date)['Adj Close']
        _df.name = self.benchmark

        # TODO: What if no stock data for the beginning?
        return _df

    def _compute_values(self, i: int) -> float:
        values = (self.prices.iloc[i] * self.shares.iloc[i - 1]).sum() + self.cashes[i - 1]

        return values

    def _compute_shares(self, i: int) -> pd.Series:
        shares = np.floor(self.portfolio_values[i] * np.array(self.weights) / self.prices.iloc[i])

        return shares

    def _compute_cash(self, i: int) -> float:
        cash = self.portfolio_values[i] - (self.shares.iloc[i] * self.prices.iloc[i]).sum()
        return cash

    def _compute_weights(self, i: int | str) -> list:
        if isinstance(i, int):
            curr_vals = self.shares.iloc[i] * self.prices.iloc[i]
        else:
            curr_vals = self.shares.loc[i] * self.prices.loc[i]

        return list(curr_vals / self.get_value(i))

    def _check_weights(self) -> bool:
        return sum(self.weights) == 1

    def _reset_weights(self) -> None:
        self.weights = [1 / len(self.tickers) for t in self.tickers]

# def rebalance_portfolio2(self) -> None:
#     # return: the daily value of the portfolio from start to end
#     stock_data = self.prices
#     stock_data = stock_data.reindex(columns=self.tickers)  # ! Key to mantain structure
#
#     shares_df = pd.DataFrame(index=stock_data.index, columns=self.tickers)
#     capitals = [-1] * len(self.prices.index)
#     cashes = [-1] * len(self.prices.index)
#     ind = 0
#
#     for i in stock_data.index:
#         if ind == 0:
#             # initial set up
#             capitals[ind] = self.cash
#             for s, w in zip(self.tickers, self.weights):
#                 shares_df.loc[i, s] = np.floor(capitals[0] * np.array(w) / stock_data[s].loc[i])
#
#             cashes[ind] = self.cash - (shares_df.loc[i] * stock_data.loc[i]).sum()
#             ind += 1
#             continue
#
#         capitals[ind] = (stock_data.iloc[ind] * shares_df.iloc[ind - 1]).sum() + cashes[ind - 1]
#
#         if ind % 30 == 0:
#             curr_shares = np.floor(capitals[ind] * np.array(self.weights) / stock_data.loc[i])
#             shares_df.loc[i] = curr_shares
#         else:
#             shares_df.loc[i] = shares_df.iloc[ind - 1]
#
#         cashes[ind] = capitals[ind] - (shares_df.loc[i] * stock_data.loc[i]).sum()
#         ind += 1
#
#     portfolio_values = pd.DataFrame(capitals, columns=['Value'], index=self.prices.index)
#
#     self.portfolio_values = portfolio_values
#     self.cashes = pd.DataFrame(data=cashes, columns=['Cash'], index=self.prices.index)
#     self.shares = shares_df
