import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance as yf
from matplotlib import pyplot as plt
import seaborn as sns

from finance_utils.types import Interval

"""
Returns, volatility, drawdowns, and all other % values are calculated in decimal form

Visualisations and results display % values in the % form
"""

TRADING_DAYS: int = Interval.TRADING_DAYS.value
TRADING_WEEKS: int = Interval.TRADING_WEEKS.value
TRADING_MONTHS: int = Interval.TRADING_MONTHS.value
TRADING_QUARTER: int = Interval.TRADING_QUARTER.value
TRADING_YEARS: int = Interval.TRADING_YEAR.value

def set_interval(interval: Interval):
    if interval == Interval.TRADING_DAYS:
        period = TRADING_DAYS
    elif interval == Interval.TRADING_WEEKS:
        period = TRADING_WEEKS
    elif interval == Interval.TRADING_MONTHS:
        period = TRADING_MONTHS
    elif interval == Interval.TRADING_QUARTER:
        period = TRADING_QUARTER
    elif interval == Interval.TRADING_YEAR:
        period = TRADING_YEARS
    else:
        print(interval)
        raise ValueError('Does not support this kind of freq')

    return period

# assumes we have a dataframe of returns:
def get_alpha_beta(returns: pd.Series, benchmark: str = 'SPY', benchmark_df=None) -> (float, float, float):
    start_date = returns.index[0]
    end_date = returns.index[-1]

    if benchmark_df is None:
        benchmark_p = yf.download(benchmark, start=start_date, end=end_date)
        beta, alpha, r, _, _ = stats.linregress(benchmark_p['Adj Close'].pct_change(), returns)
    else:
        beta, alpha, r, _, _ = stats.linregress(benchmark_df.pct_change(), returns)

    return alpha, beta, r

def compute_return(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().fillna(0)

def get_cumulative_return(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return np.cumprod(1 + returns) - 1

def get_rolling_volatility(returns: pd.DataFrame | pd.Series, freq: Interval, windows: int = 30) -> pd.DataFrame | pd.Series:
    temp_df = returns.dropna(axis=0)
    period = set_interval(freq)

    return temp_df.rolling(windows).std(ddof=1) * np.sqrt(period)

# ---- return type: single value ----
def get_volatility(returns: pd.Series, freq: Interval) -> float:
    temp_df = returns.dropna(axis=0)
    period = set_interval(freq)

    return temp_df.std(ddof=1) * np.sqrt(period)


def get_downside_volatility(returns: pd.Series, freq: Interval , threshold: int | float = 0) -> float:
    returns = returns.dropna(axis=0)
    n = returns.shape[0]
    downside_vol = np.sqrt(sum([min(r_i - threshold, 0) ** 2 for r_i in returns]) / (n - 1))
    period = set_interval(freq)

    return downside_vol * np.sqrt(period)


def get_annual_return(returns: pd.Series, freq: Interval, geo: bool = True) -> float:
    """
    :param geo:
    :param returns: the returns of the stock with a frequency freq
    :param freq: D | M | Q
    :return: gives out the compound annual return

    """
    period = set_interval(freq)

    returns = returns.dropna(axis=0)
    time_count = returns.shape[0] - 1

    cum_return = np.cumprod(1 + returns)

    if geo:
        annual_return = cum_return.iloc[-1] ** (period / time_count) - 1
    else:
        annual_return = cum_return.iloc[-1] * period / time_count

    return annual_return


def monthly_volatility(df: pd.Series) -> pd.Series:
    """
    :param df: pd.Series, the return of the stock
    :return: pd.Series, the monthly volatility of the stock
    """
    return ((df * 100).resample('ME').std(ddof=1) * np.sqrt(TRADING_DAYS)).rename('Monthly Volatility')


def monthly_return(df: pd.Series) -> pd.Series:
    """
    :param df: pd.Series, the price of the stock
    :return: pd.Series, the monthly return of the stock
    """
    first = df.resample('ME').first()
    monthly = (df.resample('ME').last() - first) / first

    return monthly.rename("Monthly Return").fillna(0)


def get_monthly_stats(returns: pd.Series) -> pd.DataFrame:
    monthly_df = pd.DataFrame(returns.copy())

    # ---- return ----
    def q1(x):
        return x.quantile(0.25)

    def q3(x):
        return x.quantile(0.75)

    f = {'Monthly Return': ['mean', 'std', 'median', q1, q3, 'max', 'min']}
    monthly_stats = monthly_df.groupby(monthly_df.index.month).agg(f)
    monthly_stats.index.name = 'Month'
    monthly_stats.columns = ['mean', 'std', 'median', 'q1', 'q3', 'max', 'min']
    monthly_stats = monthly_stats * 100

    return monthly_stats


def plot_monthly_box(df: pd.Series) -> None:
    dfs = [_df for _df in df.groupby(df.index.month)]
    array_of_df = []

    vals, names, xs = [], [], []

    for _df in dfs:
        new_df: pd.Series = _df[1] * 100
        new_df.name = str(_df[0])
        vals.append(new_df.values.tolist())
        names.append(new_df.name)
        xs.append([int(new_df.name) for i in range(len(new_df.index))])

    all_mean = df.mean()

    fig, ax = plt.subplots(figsize=(12, 9))
    plt.plot([i + 1 for i in range(12)], [all_mean for i in range(12)], alpha=0.2
             , linestyle='--', color='grey', label='mean')
    plt.boxplot(vals, tick_labels=names)
    for x, val in zip(xs, vals):
        plt.scatter(x, val, alpha=0.4)

    plt.title('Monthly Analysis')
    plt.xlabel('Month')
    plt.ylabel('%')
    plt.legend()
    plt.show()


def plot_heatmap(_monthly_df: pd.Series, annot: bool = True):
    """
    :param annot:
    :param _monthly_df: monthly return | monthly volatility
    :return: a heatmap
    """

    # # -- check input --
    # if _monthly_df.name == 'Monthly Return':
    #

    # -- format check --
    _monthly_df = num_to_percent(_monthly_df)

    # -- reshaping --
    _df = pd.DataFrame(_monthly_df)
    _df['Year'] = _df.index.year
    _df['Month'] = _df.index.month
    _df.index = [i for i in range(_df.shape[0])]

    # -- drawing the heatmap --
    result = _df.pivot(index='Year', columns='Month', values=_monthly_df.name)
    fig, ax = plt.subplots(figsize=(10, 8))

    ax = sns.heatmap(result, linewidths=0.30, annot=annot, center=0)
    plt.title(f"{_monthly_df.name} (%)")
    plt.show()


def yearly_return(df: pd.Series) -> pd.Series:
    """
    :param df: pd.Series, the price of the stock
    :return: pd.Series, the yearly return of the stock
    """
    try:
        yearly = (df.resample('YE').last() - df.resample('YE').first()) / df.resample('YE').first()
    except:
        try:
            yearly = (df.resample('Y').last() - df.resample('Y').first()) / df.resample('Y').first()
        except:
            raise ValueError('Error with the sample freq')

    return yearly.rename("Yearly Return").fillna(0)


def plot_yearly_return(_yearly_return: pd.Series):
    """
    :param _yearly_return: either daily price or yearly return of the stock
    :return: a bar plot of yearly return of the stock
    """

    # -- check input --
    if _yearly_return.name != 'Yearly Return':
        _yearly_return = yearly_return(_yearly_return)

    # -- format check --
    _yearly_return = round(num_to_percent(_yearly_return), 2)

    # -- plotting --
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(_yearly_return.index.year, _yearly_return)
    ax.set(xlabel='Year', ylabel='Return', title='Yearly Return (%)')
    ax.bar_label(bars)

    plt.show()


def get_sharpe_ratio(returns: pd.Series, freq: Interval, r_f: float | int = 0) -> float:
    """
    :param freq:
    :param returns: the asset returns
    :param r_f: risk-free rate
    """
    std_r_p = get_volatility(returns, freq)
    annual_r_p = get_annual_return(returns, freq) - r_f

    return annual_r_p / std_r_p


def get_sortino_ratio(returns: pd.Series, freq: Interval, r_f: float | int = 0) -> float:
    r_f = num_to_percent(r_f)

    down_vol = get_downside_volatility(returns, freq)
    annual_r_p = get_annual_return(returns, freq) - r_f

    return annual_r_p / down_vol


def get_VaR(returns: pd.Series, alpha: float = 99, lookback_days: int = None) -> float:
    returns = returns.dropna()
    if lookback_days is None:
        lookback_days = returns.shape[0]

    returns = returns.iloc[-lookback_days:]

    return np.percentile(returns, 100 - alpha)


def get_CVaR(returns: pd.Series, alpha: float = 99, lookback_days: int = None) -> float:
    returns = returns.dropna()
    if lookback_days is None:
        lookback_days = returns.shape[0]

    var_alpha = get_VaR(returns, alpha, lookback_days)

    return np.nanmean(returns[returns < var_alpha])


# ---- formats ----
def percent_to_num(x):  # converts returns to num
    # check instance
    if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
        if max(abs(x)) < 1:
            return x
        else:
            return x / 100

    elif isinstance(x, int) or isinstance(x, float):
        if abs(x) < 1:
            return x
        else:
            return x / 100

    else:
        raise TypeError(f'does not support this type of input: {type(x)}')


def num_to_percent(x):
    return 100 * percent_to_num(x)

