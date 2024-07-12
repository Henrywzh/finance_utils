import pandas as pd
import yfinance as yf


"""

class Data:
- olhcv for every tickers and benchmark (ORIGINAL DATA)
- returns, prices for every tickers
- results for every tickers

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
    def __init__(self, df: pd.DataFrame, benchmark: str):
        """"
        :param df: contains tickers & benchmark data
        :param benchmark:
        """
        formats = {'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'}
        if set(df.columns.get_level_values(0).tolist()) == formats:
            self.tickers = list(set(df.columns.get_level_values(1).tolist()))
        else:
            raise ValueError('Wrong Format')
        self.benchmark = benchmark
        self.weights = [1 / len(self.tickers) for i in self.tickers]

        # need to check column format


if __name__ == '__main__':
    df = yf.download(tickers=['AAPL', 'MSFT', 'AMZN', 'SPY'], start='2020-01-01')
    p = Portfolio(df, 'SPY')
    print(df)
