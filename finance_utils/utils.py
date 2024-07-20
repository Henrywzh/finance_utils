import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance as yf
from matplotlib import pyplot as plt
import seaborn as sns

# ---- Functions ----
"""
functions for all of below:

rolling: [downside beta, upside beta, beta, alpha, ***
          downside volatility, upside volatility, volatility skewness = upside var / downside var, 
          returns skewness & kurtosis, -> just use skew() & kurt()
          volatility, returns, prices, ***
          sharpe ratio, **
          sortino ratio, **
          var95, var99, cvar95, cvar99, -> require returns ***
          annualised returns(geo + ari)]
          
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
def get_volatility(returns: pd.Series):
    if returns.abs().max() < 1:
        temp_df = returns * 100
    else:
        temp_df = returns.copy()

    return temp_df.var()


def get_risk(returns: pd.Series):
    return np.sqrt(get_volatility(returns))


def get_annual_return(returns: pd.Series, freq: str = 'D') -> float:
    """
    :param returns: the returns of the stock with a frequency freq
    :param freq: D | M | Q
    :return: gives out the compound annual return

    """
    if freq == 'D':
        period = 252
    elif freq == 'M' or freq == 'ME':
        period = 12
    elif freq == 'Q':
        period = 4
    else:
        raise ValueError('Does not support this kind of freq')

    time_count = returns.shape[0]
    returns = percent_to_num(returns)

    return (np.cumprod(1 + returns) - 1).iloc[-1] ** (period / time_count)


def monthly_return(df: pd.Series) -> pd.Series:
    """
    :param df: pd.Series, the price of the stock
    :return: pd.Series, the monthly return of the stock
    """
    monthly = (df.resample('ME').last() - df.resample('ME').first()) / df.resample('ME').first()

    return monthly.rename("Monthly Return")


def plot_return_heatmap(_monthly_return: pd.Series):
    """
    :param _monthly_return: either daily price or monthly return of the stock
    :return: a heatmap
    """

    # -- check input --
    if _monthly_return.name != 'Monthly Return':
        _monthly_return = monthly_return(_monthly_return)

    # -- reshaping --
    _df = pd.DataFrame(_monthly_return)
    _df['Year'] = _df.index.year
    _df['Month'] = _df.index.month
    _df.index = [i for i in range(_df.shape[0])]

    # -- drawing the heatmap --
    result = _df.pivot(index='Year', columns='Month', values='Monthly Return')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.heatmap(result, linewidths=0.30, annot=True)
    plt.title("Calendar Return")


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

    _yearly_return = round(_yearly_return, 3)

    # -- plotting --
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(_yearly_return.index.year, _yearly_return)
    ax.set(xlabel='Year', ylabel='Return', title='Yearly Return')
    ax.bar_label(bars)


def get_sharpe_ratio(returns: pd.Series, r_f: float | int = 0) -> float:
    """
    :param returns: the asset returns
    :param r_f: risk-free rate
    """
    std_r_p = get_risk(returns)
    annual_r_p = get_annual_return(returns) - r_f

    return annual_r_p / std_r_p


def get_sortino_ratio(returns: pd.Series, r_f: float | int = 0) -> float:
    n = returns.shape[0]
    down_vol = sum([max(r_i - r_f, 0) ** 2 for r_i in returns]) / (n - 1)
    annual_r_p = get_annual_return(returns) - r_f

    return annual_r_p / np.sqrt(down_vol)


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
def percent_to_num(returns):  # converts returns to num
    if max(abs(returns)) < 1:
        return returns
    else:
        return returns / 100


def num_to_percent(returns):
    return 100 * percent_to_num(returns)


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
