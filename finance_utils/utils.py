import numpy as np
import pandas as pd
import scipy.stats as stats

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
def get_alpha_beta(returns: pd.DataFrame, ticker: str, benchmark: str) -> (float, float, float):
    beta, alpha, r, _, _ = stats.linregress(returns[ticker], returns[benchmark])
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


def get_annual_return(returns: pd.Series):
    """
    :param returns:
    :return: gives out the compound annual return

    if want calendar year performance, input returns with data constrained only within the year
    """
    days = returns.shape[0]
    returns = percent_to_num(returns)
    # dates?
    # assumes daily

    return (np.cumprod(1 + returns) - 1) ** (365 / days)


def get_sharpe_ratio(returns: pd.Series, r_f: float | int = 0) -> float:
    """
    :param returns: the asset returns
    :param r_f: risk-free rate


    TODO: Allow user to choose r_f?
    """

    std_r_p = get_risk(returns)
    annual_r_p = get_annual_return(returns) - r_f
    return annual_r_p / std_r_p


def get_sortino_ratio(returns: pd.Series, r_f: float | int = 0) -> float:
    """
    TODO: Same as sharpe_ratio
    """
    n = returns.shape[0]
    down_vol = sum([max(r_i - r_f, 0) ** 2 for r_i in returns]) / (n - 1)
    annual_r_p = get_annual_return(returns) - r_f
    return annual_r_p / np.sqrt(down_vol)


def get_VaR(returns: pd.Series, alpha: float = 99, lookback_days: int = None) -> float:
    returns = returns.dropna()
    if lookback_days is None:
        lookback_days = returns.shape[0]

    returns = returns.iloc[-lookback_days:]
    return np.percentile(returns, 100 * (1 - alpha))


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
    return x.mean()


def var(x):
    return x.var()


def std(x):
    return x.std()


def skew(x):
    return stats.skew(x)


def kert(x):
    return stats.kurtosis(x)
