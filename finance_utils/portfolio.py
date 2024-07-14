import pandas as pd
import yfinance as yf


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
            df: pd.DataFrame = None,
            benchmark: str = None,
            start_date: str = None,
            end_date: str = None
            ):
        """"
        :param df: contains tickers & benchmark data
        :param benchmark:
        """
        self.start_date = '2020-01-01' if start_date is None else start_date
        self.end_date = '2024-07-01' if end_date is None else end_date
        self.df = df
        # TODO: Check df format
        self.tickers = []
        self.benchmark = benchmark
        self.weights = []

    def set_start_date(self, start_date: str):
        self.start_date = start_date

    def set_end_date(self, end_date: str):
        self.end_date = end_date

    def add(self, item: str | list[str]):
        self.tickers += item
        is_multi_index = False if isinstance(item, str) else True
        new_item = yf.download(item, self.start_date, self.end_date)

        # need to check column format


if __name__ == '__main__':
    df = yf.download(tickers=['AAPL', 'MSFT', 'AMZN', 'SPY'], start='2020-01-01')
    p = Portfolio(df, 'SPY')
    print(df)

