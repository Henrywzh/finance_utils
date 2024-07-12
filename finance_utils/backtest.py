import pandas as pd


class Data:
    def __init__(self, df: pd.DataFrame):
        self.df = df  # ohlcv
        self.prices = df['Adj Close' if 'Adj Close' in df.columns.get_level_values(0).tolist() else 'Close']
        self.returns = self.prices.pct_change()  # contains na values
        self.results = self.returns.copy()


class Backtest:
    def __init__(self, data: Data, start_date):
        self.results = data  # assumes results initially contains the returns & price
        self.start_date = start_date
