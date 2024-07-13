from utils import *


class Backtest:
    def __init__(self, data: pd.DataFrame, start_date, end_date):
        """
        :param data: initial columns: Value, Returns
        :param start_date:
        :param end_date:
        """
        if 'Value' not in data.columns or 'Return' not in data.columns:
            raise ValueError('data columns should contain: Value, Return')

        if start_date > end_date:
            raise ValueError('start_date should be less than end_date')

        if start_date is None or start_date < data.index[0]:
            start_date = data.index[0]

        if end_date is None or end_date > data.index[-1]:
            end_date = data.index[-1]

        self.df = data  # assumes results initially contains the returns & Value
        self.start_date = start_date
        self.end_date = end_date
        self.results = dict()  # contains all the single result values, eg max drawdown...

        self.run()

    # ---- The main function ----
    def run(self) -> None:  # run the backtest
        pass

    def get_results(self) -> dict:  # get results after run
        return self.results

    def get_df(self) -> pd.DataFrame:
        return self.df

    def plot(self):
        pass
        # TODO: Plot the backtest result

    # ---- Single value functions ----
    def initial_value(self) -> float:
        return self.df['Value'].first()

    def peak_value(self) -> float:
        return self.df['Value'].max()

    def final_value(self) -> float:
        return self.df['Value'].last()

    def max_drawdown(self) -> float:  # some issues with signs
        return -(self.df['Drawdown']).min()

    def avg_drawdown(self) -> float:  # some issues with signs
        return -(self.df['Drawdown'][self.df['Drawdown'] < 0]).mean()

    def calmar_ratio(self) -> float:
        annualised_return = get_annual_return(self.df['Return'])
        max_drawdown = self.max_drawdown()
        return annualised_return / max_drawdown

    def sterling_ratio(self) -> float:
        annualised_return = get_annual_return(self.df['Return'])
        avg_drawdown = self.avg_drawdown()
        return annualised_return / avg_drawdown

    def annualised_return(self) -> float:
        return get_annual_return(self.df['Return'])

    # TODO: Store these values into the dict

    # ---- time series value functions ----
    def peak(self) -> pd.Series:
        peaks = pd.Series([self.df['Value'].iloc[:i + 1].max() for i in range(self.df.shape[0])], index=self.df.index)
        return peaks

    def drawdown(self) -> pd.Series:
        peaks = self.peak()
        return self.df['Value'] / peaks - 1

    """
    peak, drawdown, avg drawdown, max drawdown, **
    calmar ratio, sterling ratio, ***
    annualised returns, ***
    exposure time, max/avg drawdown duration, 

    win rate,
    best/worst/avg trade %, max/avg trade duration,
    profit factor, expectancy, SQN
    """
