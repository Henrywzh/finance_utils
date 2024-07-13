from utils import *
import yfinance as yf

class Backtest:
    def __init__(self, data: pd.DataFrame, start_date, end_date, benchmark: str = 'SPY', r_f=None):
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

        self.r_f = 0 if r_f is None else r_f
        self.benchmark = benchmark
        self.df = data  # assumes results initially contains the returns & Value
        self.start_date = start_date
        self.end_date = end_date
        self.results = dict()  # contains all the single result values, eg max drawdown...

        self.run()

    # ---- The main function ----
    def run(self) -> None:  # run the backtest
        # -- clean all stuff first --
        self.reset()

        # -- adding stuff to the df --
        self.df['Peak'] = self.peak()
        self.df['Drawdown'] = self.drawdown()

        # -- adding stuff to the results --
        self.results['Initial Value'] = self.initial_value()
        self.results['Peak Value'] = self.peak_value()
        self.results['Final Value'] = self.final_value()
        self.results['Max Drawdown'] = self.max_drawdown()
        self.results['Avg Drawdown'] = self.avg_drawdown()
        self.results['Calmar Ratio'] = self.calmar_ratio()
        self.results['Sterling Ratio'] = self.sterling_ratio()
        self.results['Annualised Return'] = self.annualised_return()
        self.results['Volatility'] = self.volatility()
        self.results['Sharpe Ratio'] = self.sharpe_ratio()
        self.results['Downside Volatility'] = self.downside_volatility()
        self.results['Sortino Ratio'] = self.sortino_ratio()
        self.results['VaR 95'] = self.value_at_risk(alpha=95)
        self.results['VaR 99'] = self.value_at_risk(alpha=99)
        self.results['CVaR 95'] = self.conditional_VaR(alpha=95)
        self.results['CVaR 99'] = self.conditional_VaR(alpha=99)
        alpha, beta, r_2 = self.alpha_beta_r(self.benchmark)
        self.results['Alpha'] = alpha
        self.results['Beta'] = beta
        self.results['R^2'] = r_2

    def get_results(self) -> dict:  # get results after run
        return self.results

    def get_df(self) -> pd.DataFrame:
        return self.df

    def plot(self):
        pass
        # TODO: Plot the backtest result

    def reset(self):
        self.results = dict()
        columns = self.df.columns.tolist()[2:]
        self.df = self.df.drop(columns=columns)
        print("df & results cleaned")

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

    def volatility(self) -> float:
        return get_volatility(self.df['Return'])

    def sharpe_ratio(self) -> float:
        return get_sharpe_ratio(self.df['Return'], self.r_f)

    def downside_volatility(self) -> float:
        return get_volatility(get_downside_returns(self.df['Return']))

    def sortino_ratio(self) -> float:
        return get_sortino_ratio(self.df['Return'], self.r_f)

    def value_at_risk(self, alpha=99, lookback_days=None) -> float:
        return get_VaR(self.df['Return'], alpha, lookback_days)

    def conditional_VaR(self, alpha=99, lookback_days=None) -> float:
        return get_CVaR(self.df['Return'], alpha, lookback_days)

    def alpha_beta_r(self, benchmark: str) -> (float, float, float):
        benchmark_df = yf.download(benchmark, start=self.start_date, end=self.end_date)
        benchmark_returns = benchmark_df['Adj Close'].pct_change()
        beta, alpha, r_2, _, _ = stats.linregress(self.df['Return'], benchmark_returns)
        return alpha, beta, r_2

    def alpha(self, benchmark: str) -> float:
        alpha, _, _ = self.alpha_beta_r(benchmark)
        return alpha

    def beta(self, benchmark: str) -> float:
        _, beta, _ = self.alpha_beta_r(benchmark)
        return beta

    # TODO: get r_f from yf directly instead of users' input

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
