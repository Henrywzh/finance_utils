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

class BackTest:
- given strategy/asset/portfolio returns
- benchmark returns
- value of your wealth
- xxx ratio, volatility...
- alpha, beta...

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
        self.df = data
        self.cash = cash
        self.tickers = data.columns.tolist()

        # set benchmark
        self.benchmark = '^GSPC' if benchmark is None else benchmark

        # set date
        self.start_date = start_date if start_date else data.index[0]
        self.end_date = end_date if end_date else data.index[-1]


        self.benchmark_price = self._download_benchmark()
        self.weights = default_weights if default_weights else [1 / len(self.tickers)] * len(self.tickers)
        self.start_date = start_date if start_date else '2015-01-01'
        self.end_date = datetime.datetime.now() if not end_date else end_date

    # -- set up, changes to portfolio settings --
    def set_start_date(self, start_date: str):
        self.start_date = start_date

    def set_end_date(self, end_date: str):
        self.end_date = end_date

    def add(self, item: str | list[str]):
        self.tickers += item
        # TODO:

        # need to check column format

    # -- quantitative analysis --
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
    def _download_benchmark(self):
        _df = yf.download(self.benchmark, self.start_date, self.end_date)
        return _df

    def _check_weights(self) -> bool:
        # TODO:
        return sum(self.weights) == 1


if __name__ == '__main__':
    df = yf.download(tickers=['AAPL', 'MSFT', 'AMZN', 'SPY'], start='2020-01-01')
    initial_capital = 1_000_000
    p = Portfolio(df['Adj Close'], initial_capital)
    print(df)

