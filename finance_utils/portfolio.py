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
            cash: int | float,
            start_date: str = None,
            end_date: str = None,
            benchmark: str = None,
            ):
        """
        :param cash:
        :param start_date:
        :param end_date:
        :param benchmark:
        """
        # -- format check ---
        if cash < 10000:
            raise ValueError("Please enter cash amount >= 10_000")

        if start_date > end_date:
            raise ValueError('start_date should be less than end_date')

        # -- initialisation --
        self.df = pd.DataFrame()
        self.asset_values = {'Cash': cash}  # (key) asset, (value) asset value
        self.tickers = ['Cash']
        self.benchmark = 'Price' if benchmark is None else benchmark
        self.weights = {'Cash': 1}  # sum of weights == 1
        self.start_date = str(start_date) if start_date else '2020-01-01'
        self.end_date = datetime.datetime.now() if end_date is None else end_date

    def set_start_date(self, start_date: str):
        self.start_date = start_date

    def set_end_date(self, end_date: str):
        self.end_date = end_date

    def add(self, item: str | list[str]):
        self.tickers += item
        is_multi_index = False if isinstance(item, str) else True
        new_item = yf.download(item, self.start_date, self.end_date)

        # need to check column format

    def get_portoflio_returns(self):
        # pass the portfolio returns as a pd.Series
        pass

    def _check_weights(self) -> bool:
        return sum(self.weights.values()) == 1


if __name__ == '__main__':
    df = yf.download(tickers=['AAPL', 'MSFT', 'AMZN', 'SPY'], start='2020-01-01')
    p = Portfolio(df)
    print(df)

