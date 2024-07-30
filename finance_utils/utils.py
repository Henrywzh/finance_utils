import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance as yf
from matplotlib import pyplot as plt
import seaborn as sns

# ---- Functions ----
"""
functions for all of below:

rolling: [downside beta, upside beta,
          downside volatility, upside volatility, volatility skewness = upside var / downside var, 
          returns skewness & kurtosis, -> just use skew() & kurt()
          ]
          
          ---- IMPORTANT ----
          For rolling data, just input the data with restricted range
          
single: rolling + 
        [peak, drawdown, avg drawdown, max drawdown, calmar ratio, sterling ratio, 
        annualised returns(geo + ari),
        exposure time, max/avg drawdown duration, win rate, 
        best/worst/avg trade %, max/avg trade duration,
        profit factor, expectancy, SQN
        ]
        
        
REMINDER: 
- downside volatility = np.sqrt(sum(max(r_i - r_f, 0)**2 / (n - 1))
- sortino ratio = (r_i - r_f) / downside volatility
"""

# ---- constants -----
TRADING_DAYS: int = 252
TRADING_MONTHS: int = 12
TRADING_QUARTER: int = 4


# assumes we have a dataframe of returns:
def get_alpha_beta(returns: pd.Series, benchmark: str = 'SPY') -> (float, float, float):
    start_date = returns.index[0]
    end_date = returns.index[-1]
    benchmark_p = yf.download(benchmark, start=start_date, end=end_date)
    beta, alpha, r, _, _ = stats.linregress(returns, benchmark_p['Adj Close'].pct_change())
    return alpha, beta, r


def get_downside_returns(returns: pd.Series, threshold=0) -> pd.Series:
    return returns[returns < threshold]


def get_upside_returns(returns: pd.Series, threshold=0) -> pd.Series:
    return returns[returns > threshold]


# ---- return type: single value ----
def get_volatility(returns: pd.Series) -> float:
    temp_df = returns.dropna(axis=0)

    return temp_df.std(ddof=1) * np.sqrt(TRADING_DAYS)


def get_downside_volatility(returns: pd.Series, threshold: int | float = 0) -> float:
    returns = returns.dropna(axis=0)
    n = returns.shape[0]
    downside_vol = np.sqrt(sum([min(r_i - threshold, 0) ** 2 for r_i in returns]) / (n - 1))

    return downside_vol * np.sqrt(TRADING_DAYS)


def get_annual_return(returns: pd.Series, freq: str = 'D', geo: bool = True) -> float:
    """
    :param geo:
    :param returns: the returns of the stock with a frequency freq
    :param freq: D | M | Q
    :return: gives out the compound annual return

    """
    freq = freq.upper()

    if freq == 'D':
        period = TRADING_DAYS
    elif freq == 'M' or freq == 'ME':
        period = TRADING_MONTHS
    elif freq == 'Q':
        period = TRADING_QUARTER
    else:
        raise ValueError('Does not support this kind of freq')

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
    monthly = (df.resample('ME').last() - df.resample('ME').first()) / df.resample('ME').first()

    return monthly.rename("Monthly Return")


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
    fig, ax = plt.subplots(figsize=(10, 6))

    ax = sns.heatmap(result, linewidths=0.30, annot=annot, center=0)
    plt.title(f"{_monthly_df.name} (%)")
    plt.show()


def yearly_return(df: pd.Series) -> pd.Series:
    """
    :param df: pd.Series, the price of the stock
    :return: pd.Series, the yearly return of the stock
    """
    yearly = (df.resample('YE').last() - df.resample('YE').first()) / df.resample('YE').first()

    return yearly.rename("Yearly Return")


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


def get_sharpe_ratio(returns: pd.Series, r_f: float | int = 0) -> float:
    """
    :param returns: the asset returns
    :param r_f: risk-free rate
    """
    std_r_p = get_volatility(returns)
    annual_r_p = get_annual_return(returns) - r_f

    return annual_r_p / std_r_p


def get_sortino_ratio(returns: pd.Series, r_f: float | int = 0) -> float:
    r_f = num_to_percent(r_f)

    down_vol = get_downside_volatility(returns)
    annual_r_p = get_annual_return(returns) - r_f

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


# ---- stats ----
def mean(x):
    return np.mean(x)


def var(x):
    return np.var(x, ddof=1)


def std(x):
    return np.std(x, ddof=1)


def skew(x):
    return stats.skew(x)


def kert(x):
    return stats.kurtosis(x)
