import pandas as pd
import yfinance as yf
import datetime


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

        if start_date > end_date:
            raise ValueError("start_date should be less than end_date")

        if len(df.columns) != len(default_weights):
            raise ValueError("number of stocks != length of default_weights")

        # -- initialisation --
        self.shares = pd.DataFrame()
        self.prices = data
        self.cash = cash
        self.cash_df = pd.DataFrame()

        self.tickers = data.columns.tolist()

        # set benchmark
        self.benchmark = '^GSPC' if benchmark is None else benchmark

        # set date
        self.start_date = start_date if start_date else data.index[0]
        self.end_date = end_date if end_date else data.index[-1]

        self.benchmark_price = self._get_benchmark()
        self.weights = default_weights if default_weights else [1 / len(self.tickers) for t in self.tickers]
        self.start_date = start_date if start_date else '2015-01-01'
        self.end_date = datetime.datetime.now() if not end_date else end_date

    # -- run --
    def run(self) -> pd.DataFrame:
        # return: the daily value of the portfolio from start to end
        # TODO:
        pass

    # -- set up, changes to portfolio settings --
    def set_start_date(self, start_date: str) -> None:
        self.start_date = start_date

    def set_end_date(self, end_date: str) -> None:
        self.end_date = end_date

    def add_stock(self, item: str | list[str]) -> None:
        if isinstance(item, str):
            self.tickers += [item]
        else:
            self.tickers += item

        self._reset_weights()
        # TODO:

        # need to check column format

    # -- quantitative analysis --
    def get_value(self, i: int | str) -> float:
        stocks_val = self.shares * self.prices.iloc[i] if isinstance(i, int) else self.shares * self.prices.loc[i]
        total_val = self.cash + stocks_val.sum()
        return total_val

    def get_portfolio_values(self) -> pd.DataFrame:
        # TODO:
        pass

    def get_portfolio_returns(self):
        # pass the portfolio returns as a pd.Series
        # TODO:
        pass

    def get_portfolio_risk(self):
        # TODO:
        pass

    # -- portfolio management --
    def rebalance_portfolio(self):
        # TODO:
        pass

    def optimise_portfolio(self):
        # TODO:
        pass

    # -- private methods --
    def _download(self, ticker: str | list[str]) -> None:
        _df = yf.download(ticker, self.start_date, self.end_date)
        self.prices = self.prices.merge(_df, how='left')
        # TODO: Debug

    def _get_benchmark(self) -> pd.DataFrame:
        _df = yf.download(self.benchmark, self.start_date, self.end_date)
        # TODO: What if no stock data for the beginning?
        return _df

    def _compute_weights(self, i: int | str) -> list:
        curr_vals = self.shares * self.prices.iloc[i] if isinstance(i, int) else self.shares * self.prices.loc[i]
        return list(curr_vals / self.get_value(i))

    def _check_weights(self) -> bool:
        return sum(self.weights) == 1

    def _reset_weights(self) -> None:
        self.weights = [1 / len(self.tickers) for t in self.tickers]


if __name__ == '__main__':
    df = yf.download(tickers=['AAPL', 'MSFT', 'AMZN', 'SPY'], start='2020-01-01')
    initial_capital = 1_000_000
    p = Portfolio(df['Adj Close'], initial_capital)
    print(df)

