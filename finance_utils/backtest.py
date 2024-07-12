from datetime import datetime

import pandas as pd
import yfinance as yf
from utils import *


# class Data:
#
#
# class Results(Data):
#     def __init__(self, df: pd):
#         self.df = df
#
#
# class RawData(Data):
#     def __init__(self, df: pd):
#         self.df = df
#
#


class Backtest:
    def __init__(self, data: pd.DataFrame, start_date, end_date):
        """
        :param data: initial columns: Value, Returns
        :param start_date:
        :param end_date:
        """
        if 'Value' not in data.columns or 'Return' not in data.columns:
            raise ValueError('data columns should contain: Value, Return')

        self.df = data  # assumes results initially contains the returns & Value
        self.start_date = start_date
        self.end_date = end_date
        self.results = dict()  # contains all the single result values, eg max drawdown...

    # ---- The main function ----
    def run(self):  # run the backtest
        pass

    def get_results(self):  # get results after run
        return self.results

    # ---- Single value functions ----
    def initial_value(self) -> float:
        return self.df['Value'].first()

    def peak_value(self) -> float:
        return self.df['Value'].max()

    def final_value(self) -> float:
        return self.df['Value'].last()

    def max_drawdown(self):  # some issues with signs
        return -(self.df['Drawdown']).min()

    def avg_drawdown(self):  # some issues with signs
        return -(self.df['Drawdown'][df['Drawdown'] < 0]).mean()

    def calmar_ratio(self):
        annualised_return = get_annual_return(self.df['Return'])
        return

    # TODO: Store these values into the dict


    # ---- time series value functions ----
    # TODO: Maybe change the return type? User can get the pd.Series of the results below
    def peak(self):
        self.df['Peak'] = [self.df['Value'].iloc[:i + 1].max() for i in range(self.df.shape[0])]

    def drawdown(self):
        self.df['Drawdown'] = self.df['Value'] / self.df['Peak'] - 1

    """
    peak, drawdown, avg drawdown, max drawdown, **
    calmar ratio, sterling ratio,
    annualised returns(geo + ari),
    exposure time, max/avg drawdown duration,

    win rate,
    best/worst/avg trade %, max/avg trade duration,
    profit factor, expectancy, SQN
    """


if __name__ == '__main__':
    df = yf.download('TSLA', start=datetime(2020, 1, 1))
