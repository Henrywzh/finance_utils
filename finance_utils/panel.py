from .indicators import *

class Panel:
    def __init__(self, tickers: list, df: pd.DataFrame):
        l = df.index.values.tolist()
        self.df = df
        self.start_time = l[0]
        self.end_time = l[-1]
        self.num_of_assets = len(tickers)

    def macd(self, ticker: str, elem: str):
        _df = self.df[(elem, ticker)].copy()


