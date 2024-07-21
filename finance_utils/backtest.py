from .utils import *
import yfinance as yf


class Backtest:
    def __init__(
            self,
            data: pd.DataFrame,
            start_date: str = None,
            end_date: str = None,
            benchmark: str = None,
            r_f: float | int = None
    ):
        """
        :param data: initial columns: Value, Return, Price
        :param start_date:
        :param end_date:
        :param benchmark:
        :param r_f:
        """

        # TODO: Buy & Hold Return | Benchmark Return?
        # TODO: Change all the format to percentage (%)

        # -- format check --
        if 'Value' not in data.columns or 'Return' not in data.columns or 'Price' not in data.columns:
            raise ValueError('data columns should contain: Value, Return, Price')

        if start_date is None or start_date < data.index[0]:
            start_date = data.index[0]

        if end_date is None or end_date > data.index[-1]:
            end_date = data.index[-1]

        if start_date > end_date:
            raise ValueError('start_date should be less than end_date')

        # -- initialisation --
        self.r_f = 0 if r_f is None else r_f
        self.benchmark = 'Price' if benchmark is None else benchmark
        self.df = data  # assumes results initially contains the returns & Value
        self.start_date = start_date
        self.end_date = end_date
        self.results = dict()  # contains all the single result values, eg max drawdown...

        self.run()

    # ---- The main function ----
    def run(self) -> None:  # run the backtest
        print("Running the backtest...")

        # TODO: Add customise functions, allow users to choose which item to be included

        # -- clean all stuff first --
        self.reset()

        # -- adding stuff to the df --
        self.df['Benchmark'] = self._get_benchmark()
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

        alpha, beta, r_2 = self.alpha_beta_r()
        self.results['Alpha'] = alpha
        self.results['Beta'] = beta
        self.results['R^2'] = r_2

        self.results['Calendar Month Return'] = self.calendar_month_return()
        self.results['Calendar Year Return'] = self.calendar_year_return()

        print("Backtesting completed")

    def get_results(self) -> dict:  # get results after run
        return self.results

    def get_df(self) -> pd.DataFrame:
        return self.df

    def plot(self):
        pass
        # TODO: Plot the backtest result

    # ---- Reset parameters ----
    def reset(self):
        self.results = dict()
        columns = self.df.columns.tolist()[3:]  # removes all the columns besides from (Price, Value, Return)
        self.df = self.df.drop(columns=columns)
        self.set_benchmark_to_buy_and_hold()
        print("df & results cleaned, benchmark set to buy & hold")

    def _get_benchmark(self) -> pd.Series:
        if self.benchmark == 'Price':
            return self.df['Price'].copy()

        try:
            benchmark_df = yf.download(self.benchmark, start=self.start_date, end=self.end_date)
            return benchmark_df['Adj Close']
        except:
            raise ValueError('Benchmark not found on Yahoo Finance')

    # ---- Changing attributes ----
    def set_start_date(self, start_date: str):
        # TODO: make sure the type is correct
        self.start_date = start_date
        self._check_date()

    def set_end_date(self, end_date: str):
        # TODO: make sure the type is correct
        self.end_date = end_date
        self._check_date()

    def set_benchmark(self, benchmark: str):
        self.benchmark = benchmark
        # TODO: add benchmark data to self.df

    def set_benchmark_to_buy_and_hold(self):
        self.benchmark = 'Price'

    # ---- Error Check ----
    def _check_date(self):
        if self.start_date > self.end_date:
            raise ValueError('Error: start_date > end_date')

        if self.start_date < self.df.index[0]:
            self.start_date = self.df.index[0]

        if self.end_date > self.df.index[-1]:
            self.end_date = self.df.index[-1]

    # ---- Single value functions ----
    def initial_value(self) -> float:
        return self.df['Value'].iloc[0]

    def peak_value(self) -> float:
        return self.df['Value'].max()

    def final_value(self) -> float:
        return self.df['Value'].iloc[-1]

    def max_drawdown(self) -> float:  # some issues with signs
        return -(self.df['Drawdown'].min())

    def avg_drawdown(self) -> float:  # some issues with signs
        return -(self.df['Drawdown'][self.df['Drawdown'] < 0].mean())

    def calmar_ratio(self) -> float:
        annualised_return = get_annual_return(self.df['Return'])
        max_drawdown = self.max_drawdown()
        return annualised_return / max_drawdown

    def sterling_ratio(self) -> float:
        annualised_return = get_annual_return(self.df['Return'])
        avg_drawdown = self.avg_drawdown()
        return annualised_return / avg_drawdown

    def ulcer_index(self) -> float:
        # TODO
        pass

    def martin_ratio(self) -> float:
        # TODO
        pass

    def annualised_return(self) -> float:
        return get_annual_return(self.df['Return'])

    def volatility(self) -> float:
        return get_volatility(self.df['Return'])

    def sharpe_ratio(self) -> float:
        return get_sharpe_ratio(self.df['Return'], self.r_f)

    def downside_volatility(self) -> float:
        return get_downside_volatility(self.df['Return'])

    def sortino_ratio(self) -> float:
        return get_sortino_ratio(self.df['Return'], self.r_f)

    def value_at_risk(self, alpha: float | int = 99, lookback_days: int = None) -> float:
        return get_VaR(self.df['Return'], alpha, lookback_days)

    def conditional_VaR(self, alpha: float | int = 99, lookback_days: int = None) -> float:
        return get_CVaR(self.df['Return'], alpha, lookback_days)

    def alpha_beta_r(self) -> (float, float, float):
        returns = self.df['Return'].dropna(axis=0)
        benchmark_returns = self.df['Price'].pct_change().dropna(axis=0)
        if self.benchmark == 'Price':
            beta, alpha, r_2, _, _ = stats.linregress(returns, benchmark_returns)
        else:

            benchmark_df = yf.download(self.benchmark, start=self.start_date, end=self.end_date)
            benchmark_returns = benchmark_df['Adj Close'].pct_change().dropna(axis=0)
            beta, alpha, r_2, _, _ = stats.linregress(returns, benchmark_returns)

        return alpha, beta, r_2

    def alpha(self) -> float:
        alpha, _, _ = self.alpha_beta_r()
        return alpha

    def beta(self) -> float:
        _, beta, _ = self.alpha_beta_r()
        return beta

    # TODO: get r_f from yf directly instead of users' input

    # ---- time series value functions ----
    def peak(self) -> pd.Series:
        peaks = pd.Series([self.df['Value'].iloc[:i + 1].max() for i in range(self.df.shape[0])], index=self.df.index)
        return peaks

    def drawdown(self) -> pd.Series:
        peaks = self.peak()
        return self.df['Value'] / peaks - 1

    def calendar_month_return(self) -> pd.Series:
        return monthly_return(self.df['Price'])

    def calendar_year_return(self) -> pd.Series:
        return yearly_return(self.df['Price'])

    """
    peak, drawdown, avg drawdown, max drawdown, **
    calmar ratio, sterling ratio, ***
    annualised returns, ***
    exposure time, max/avg drawdown duration, 

    win rate,
    best/worst/avg trade %, max/avg trade duration,
    profit factor, expectancy, SQN
    """
