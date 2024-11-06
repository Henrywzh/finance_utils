from .utils import *
from.types import *

class Visualise:
    def __init__(self, df: pd.DataFrame, results: dict, info: Info):
        self.df = df
        self.results = results
        self.info = info

        print("---- Backtesting completed ----\n")
        self._print_results()
        self.plot()

    # -------- Visualisation --------
    def plot(self) -> None:
        """
        the main plot function
        :return:
        """
        self.plot_cumulative_returns()
        self.plot_drawdown()
        self.plot_volatility()
        self.plot_monthly_return()
        self.plot_yearly_return()

    def plot_cumulative_returns(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(100 * self.df['Cumulative Benchmark Return'], label=self.info.BENCHMARK)
        ax.plot(100 * self.df['Cumulative Return'], label=self.info.STRATEGY)
        ax.set(xlabel='Date', ylabel='Cumulative Return (%)', title='Strategy & Benchmark Comparison')
        plt.legend()
        plt.show()

    def plot_drawdown(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.df['Drawdown'] * 100, label='Drawdown')
        ax.set(xlabel='Date', ylabel='Drawdown (%)', title='Drawdown')
        plt.legend(loc='best')
        plt.show()

    def plot_volatility(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.results['Rolling Volatility'], label='Volatility')
        ax.set(xlabel='Date', ylabel='Volatility (%)', title='Rolling Volatility')
        plt.legend(loc='best')
        plt.show()

    def plot_monthly_return(self) -> None:
        plot_heatmap(self.results['Monthly Return'])

    def plot_yearly_return(self) -> None:
        # plot_yearly_return(self.results['Yearly Return'])

        # -- format check --
        _yearly_return = round(100 * self.results['Yearly Return'], 2)
        _buy_hold_yearly_return = round(100 * yearly_return(self.df['Price']), 2)

        # -- plotting --
        fig, ax = plt.subplots(figsize=(10, 6))

        bars_buy_hold = ax.bar(_yearly_return.index.year, _buy_hold_yearly_return, label='Buy & Hold', alpha=0.5)
        ax.bar_label(bars_buy_hold)

        bars_strategy = ax.bar(_yearly_return.index.year, _yearly_return, label='Strategy')
        ax.bar_label(bars_strategy)

        ax.set(xlabel='Year', ylabel='Return', title='Yearly Return (%)')
        ax.legend(loc='best', ncols=2)

        plt.show()

    def plot_monthly_volatility(self) -> None:
        plot_heatmap(self.results['Monthly Volatility'])

    def plot_rolling(self) -> None:
        self.plot_rolling_VaR()
        self.plot_rolling_beta()
        self.plot_rolling_alpha()

    def plot_rolling_VaR(self) -> None:
        # TODO: plot both VaR and CVaR
        pass

    def plot_rolling_beta(self) -> None:
        pass

    def plot_rolling_alpha(self) -> None:
        pass

    def _print_results(self) -> None:
        for key in self.results:
            if isinstance(self.results[key], pd.Series) or isinstance(self.results[key], pd.DataFrame):
                continue

            print(f'{key}: {self.results[key]}')


class Result:
    def __init__(self, df: pd.DataFrame, info: Info, visualise: bool = True):
        """
        :param df: columns := ['Price', 'Value', 'Return', 'Benchmark Return']
        :param info:  keys := ['r_f' (risk-free rate), 'txn_f' (transaction fees), 'inf_r' (inflation rate),
                               'fx' (currency), 'strategy' (name of the strategy), 'benchmark' (name of the benchmark),
                               'interval' (day? week? month? year?)]
        TODO: info
        """
        self.df = df
        self.is_buy_and_hold: bool = (info.STRATEGY == StrategyType.BUY_AND_HOLD)

        # -- unpacking info --
        self.r_f: float    = info.RISK_FREE_RATE
        self.interval: int = info.INTERVAL.value

        self.info: Info    = info
        self.results: dict   = dict()
        self.visualise: bool = visualise

        self.add_to_df()
        self.add_to_results()
        self.df['Value'] = self.df['Value'].fillna(0)

        self.visual = None

        if self.visualise:
            self.visual = Visualise(self.df, self.results, self.info)

    def get_results(self):
        return self.results

    def get_df(self):
        return self.df

    def add_strategy_results(self) -> None:
        """
        private function, add strategy results to self.results
        :return:
        """
        alpha, beta, r_2 = self.alpha_beta_r()
        self.results['Alpha'] = alpha
        self.results['Beta'] = beta
        self.results['R^2'] = r_2

    def add_to_df(self):
        self.df['Peak'] = self.peak()
        self.df['Drawdown'] = self.drawdown()
        self.df['Cumulative Return'] = get_cumulative_return(self.df['Return'])
        self.df['Cumulative Benchmark Return'] = get_cumulative_return(self.df['Benchmark Return'])

    def add_to_results(self):
        self.results['Annualised Return (Geo)'] = self.annualised_return()
        self.results['Avg Annual Return (Ari)'] = self.avg_annual_return()
        self.results['Volatility (Std)'] = self.volatility()
        self.results['Sharpe Ratio'] = self.sharpe_ratio()
        self.results['Downside Volatility'] = self.downside_volatility()
        self.results['Sortino Ratio'] = self.sortino_ratio()

        self.results['VaR 99'] = self.value_at_risk(self.df['Return'], alpha=99)
        self.results['VaR 99 (Year)'] = self.value_at_risk(self.calendar_year_return(), alpha=99)
        self.results['CVaR 99'] = self.conditional_VaR(self.df['Return'], alpha=99)
        self.results['CVaR 99 (Year)'] = self.conditional_VaR(self.calendar_year_return(), alpha=99)

        self.results['Initial Value'] = self.initial_value()
        self.results['Peak Value'] = self.peak_value()
        self.results['Final Value'] = self.final_value()
        self.results['Max Drawdown'] = self.max_drawdown()
        self.results['Avg Drawdown'] = self.avg_drawdown()
        self.results['Calmar Ratio'] = self.calmar_ratio()
        self.results['Sterling Ratio'] = self.sterling_ratio()

        self.results['Monthly Return'] = self.calendar_month_return()
        self.results['Yearly Return'] = self.calendar_year_return()
        self.results['Monthly Stats'] = self.monthly_stats()

        self.results['Rolling Volatility'] = self.rolling_volatility()

        if not self.is_buy_and_hold:
            self.add_strategy_results()

    # -------- Single value functions --------
    def initial_value(self) -> float:
        return self.df['Value'].fillna(0).iloc[0]

    def peak_value(self) -> float:
        return self.df['Value'].fillna(0).max()

    def final_value(self) -> float:
        return self.df['Value'].iloc[-1]

    def max_drawdown(self) -> float:  # some issues with signs
        return -(self.df['Drawdown'].min())

    def avg_drawdown(self) -> float:  # some issues with signs
        return -(self.df['Drawdown'][self.df['Drawdown'] < 0].mean())

    def calmar_ratio(self) -> float:
        annualised_return = get_annual_return(self.df['Return'], self.info.INTERVAL)
        max_drawdown = self.max_drawdown()
        return annualised_return / max_drawdown

    def sterling_ratio(self) -> float:
        annualised_return = get_annual_return(self.df['Return'], self.info.INTERVAL)
        avg_drawdown = self.avg_drawdown()
        return annualised_return / avg_drawdown

    def ulcer_index(self) -> float:
        # TODO
        pass

    def martin_ratio(self) -> float:
        # TODO
        pass

    def annualised_return(self, geo: bool = True) -> float:
        return get_annual_return(self.df['Return'], self.info.INTERVAL, geo)

    def avg_annual_return(self) -> float:
        """
        Calculates the arithmetic average annual return by calendar year
        :return:
        """
        yearly_returns = self.calendar_year_return()

        return yearly_returns.mean()

    def volatility(self) -> float:
        return get_volatility(self.df['Return'], self.info.INTERVAL)

    def sharpe_ratio(self) -> float:
        return get_sharpe_ratio(self.df['Return'], self.info.INTERVAL, self.r_f)

    def downside_volatility(self) -> float:
        return get_downside_volatility(self.df['Return'], self.info.INTERVAL)

    def sortino_ratio(self) -> float:
        return get_sortino_ratio(self.df['Return'], self.info.INTERVAL, self.r_f)

    def value_at_risk(self, _df: pd.Series, alpha: float | int = 99, lookback_days: int = None) -> float:
        return get_VaR(_df, alpha, lookback_days)

    def conditional_VaR(self, _df: pd.Series, alpha: float | int = 99, lookback_days: int = None) -> float:
        return get_CVaR(_df, alpha, lookback_days)

    def alpha_beta_r(self) -> (float, float, float):
        alpha, beta, r_squared = get_alpha_beta(self.df['Return'], self.df['Benchmark Return'])

        return alpha, beta, r_squared

    def monthly_stats(self) -> pd.DataFrame:
        _df = self.results['Monthly Return'].copy()
        return _df.groupby(_df.index.month).describe()

    # -------- time series value functions --------
    def peak(self) -> pd.Series:
        peaks = pd.Series(
            [self.df['Value'].fillna(0).iloc[:i + 1].max() for i in range(self.df.shape[0])],
            index=self.df.index
        )
        return peaks

    def drawdown(self) -> pd.Series:
        return (self.df['Value'] / self.peak() - 1).fillna(0)

    def calendar_month_return(self) -> pd.Series:
        return monthly_return(self.df['Value'])

    def calendar_year_return(self) -> pd.Series:
        return yearly_return(self.df['Value'])

    def calendar_month_volatility(self) -> pd.Series:
        return monthly_volatility(self.df['Return'])

    def rolling_volatility(self, windows: int = 30) -> pd.Series:
        _df: pd.Series = (self.df['Return'] * 100).rolling(windows).std(ddof=1) * np.sqrt(self.interval)
        return _df.fillna(0)


"""
Backtest:
- preprocess
- return.fillna(0)
- format, dates, na values

run

reset / clear

choose

visualise

error checking
"""

class Backtest:
    def __init__(
            self,
            data: pd.DataFrame,
            visualise: bool = True,
            start_date: str = None,
            end_date: str = None,
            benchmark: str = None,
            r_f: float | int = None,
            info: Info = None,
    ):
        """
        :param data: initial column: Value (Strategy Value), Return (Strategy Return), Price (Asset Value)
        :param start_date:
        :param end_date:
        :param benchmark:
        :param r_f:
        """
        # TODO: Sliding Backtest Windows

        # TODO: data input -> benchmark and strategy
        # TODO: Consider transaction cost

        # -- format check --
        if not self._valid_data(data.columns):
            raise ValueError('data columns should contain: Value, Price')

        if start_date is None or start_date < data.index[0]:
            start_date = data.index[0]

        if end_date is None or end_date > data.index[-1]:
            end_date = data.index[-1]

        if start_date > end_date:
            raise ValueError('start_date should be less than end_date')

        # -- initialisation --
        self.r_f        = 0 if r_f is None else r_f
        self.benchmark  = 'Price' if benchmark is None else benchmark
        self.info: Info = Info() if info is None else info
        self.df         = data  # assumes results initially contains the returns & Value

        if 'Return' not in self.df.columns:
            self.df['Return'] = self.df['Value'].pct_change().fillna(0)

        if 'Benchmark Return' not in self.df.columns:
            self.df['Benchmark Return'] = self.df['Price'].pct_change().fillna(0)

        self.start_date = start_date
        self.end_date   = end_date
        self.results    = dict()  # contains all the single result values, eg max drawdown...

        self.run(visualise=visualise)

    # -------- The main function --------
    def run(self, visualise: bool = True) -> Result:  # run the backtest
        result = Result(self.df, self.info, visualise)
        self.results = result.get_results()

        return result

    # -------- Changing attributes --------

    # info: keys := ['r_f', 'txn_f', 'inf_r', 'fx', 'strategy', 'benchmark'， 'interval']
    def set_info(self, info: Info):
        self.info = info

    def set_risk_free_rate(self, r_f: float):
        self.info.set_risk_free_rate(r_f)

    def set_benchmark(self, benchmark: str):
        # TODO:
        self.benchmark = benchmark
        self.info.set_benchmark(benchmark)

    def set_transaction_free_rate(self, txn_f: float):
        self.info.set_transaction_free_rate(txn_f)

    def set_inflation_rate(self, inflation: float):
        self.info.set_inflation_rate(inflation)

    def set_strategy(self, strategy: StrategyType):
        self.info.set_strategy(strategy)

    def set_interval(self, interval: Interval):
        self.info.set_interval(interval)

    def set_start_date(self, start_date: str) -> None:
        self._valid_date(start_date, self.end_date)
        self.start_date = start_date

    def set_end_date(self, end_date: str) -> None:
        self._valid_date(self.start_date, end_date)
        self.end_date = end_date

    def set_data(self, data: pd.DataFrame) -> None:
        if not self._valid_data(data.colunmns):
            return

        self.df = data

    def set_benchmark_to_buy_and_hold(self) -> None:
        self.benchmark = 'Price'

    # -------- Error Check --------
    def _valid_data(self, column_names: list) -> bool:
        if 'Value' not in column_names or 'Price' not in column_names:
            print('data columns should contain: Value, Price')
            return False

        return True

    def _valid_date(self, start, end) -> bool:
        if start > end:
            print('Not a valid option:')
            print(f'{start} -> {end}')
            return False

        if self.start_date < self.df.index[0]:
            self.start_date = self.df.index[0]

        if self.end_date > self.df.index[-1]:
            self.end_date = self.df.index[-1]

        return True

    # TODO: Add the below indicators
    """
    exposure time, max/avg drawdown duration, 

    win rate,
    best/worst/avg trade %, max/avg trade duration,
    profit factor, expectancy, SQN
    """
