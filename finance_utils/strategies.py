import numpy as np
import pandas as pd

def fast_slow(df_prev: pd.DataFrame, fast: int, slow: int, ticker_name: str = None) -> pd.DataFrame:
    # _df contains the Close of a stock
    if fast < 0: raise ValueError('Fast must be greater than 0')
    if slow < 0: raise ValueError('Slow must be greater than 0')
    if fast > slow:
        temp = fast
        fast = slow
        slow = temp
    if ticker_name is None:
        ticker_name = 'Close'
    _df = pd.DataFrame(df_prev[ticker_name])

    _df[f'MA{fast}'] = _df[ticker_name].rolling(fast).mean()
    _df[f'MA{slow}'] = _df[ticker_name].rolling(slow).mean()

    _df['Signal'] = np.where(
        _df[f'MA{slow}'].isna(),
        0,
        np.where(_df[f'MA{slow}'] < _df[f'MA{fast}'], 1, -1)
    )  # fill na with 0, if fast MA > slow MA, signal = 1, else -1
    _df['Position'] = _df['Signal'].shift(1)
    _df['Strategy Return'] = _df[ticker_name].pct_change() * _df['Position']
    _df['Cumulative Return'] = (1 + _df['Strategy Return']).cumprod()

    # return a dataframe with columns: Price, Strategy Return, Cumulative Return
    # it also contains other stuff, but less important
    return _df


# need to redo the ema functions
def ema_fast_slow(df_prev: pd.DataFrame, fast: int, slow: int, ticker_name: str = None) -> pd.DataFrame:
    if fast < 0: raise ValueError('Fast must be greater than 0')
    if slow < 0: raise ValueError('Slow must be greater than 0')
    if fast > slow:
        temp = fast
        fast = slow
        slow = temp
    if ticker_name is None:
        ticker_name = 'Close'
    _df = pd.DataFrame(df_prev[ticker_name])

    _df[f'EMA{fast}'] = _df[ticker_name].ewm(span=fast, adjust=False).mean()
    _df[f'EMA{slow}'] = _df[ticker_name].ewm(span=slow, adjust=False).mean()

    _df['Signal'] = np.where(
        _df[f'EMA{slow}'].isna(),
        0,
        np.where(_df[f'EMA{slow}'] < _df[f'EMA{fast}'], 1, -1)
    )  # fill na with 0, if fast MA > slow MA, signal = 1, else -1
    _df['Position'] = _df['Signal'].shift(1)
    _df['Strategy Return'] = _df[ticker_name].pct_change() * _df['Position']
    _df['Cumulative Return'] = (1 + _df['Strategy Return']).cumprod()

    return _df


def buy_and_hold(df_prev: pd.DataFrame, ticker_name: str = None) -> pd.DataFrame:
    if ticker_name is None:
        ticker_name = 'Close'
    _df = pd.DataFrame(df_prev[ticker_name])

    _df['Strategy Return'] = _df[ticker_name].pct_change()
    _df['Cumulative Return'] = (1 + _df['Strategy Return']).cumprod()

    return _df


def bollinger_bands(df_prev: pd.DataFrame, period: int, step: float, ticker_name: str = None) -> pd.DataFrame:
    if period < 1: raise ValueError('Period must be >= 1')
    if step <= 0: raise ValueError('Step must be > 0')

    if ticker_name is None:
        ticker_name = 'Close'
    _df = pd.DataFrame(df_prev[ticker_name])

    _df[f'MA{period}'] = _df[ticker_name].rolling(period).mean()
    _df['Upper Band'] = _df[f'MA{period}'] + step * _df[f'MA{period}'].rolling(period).std()
    _df['Lower Band'] = _df[f'MA{period}'] - step * _df[f'MA{period}'].rolling(period).std()


# default period: 14
def rsi(df_prev: pd.DataFrame, period: int = 14, ticker_name: str = None) -> pd.DataFrame:
    if period < 1: raise ValueError('Period must be >= 1')
    if ticker_name is None:
        ticker_name = 'Close'
    _df = pd.DataFrame(df_prev[ticker_name])

    """
    RSI is more useful in trending market
    """

    _df['Diff'] = _df[ticker_name].diff()
    _df['Gain'] = np.where(_df['Diff'] > 0, _df['Diff'], 0)
    _df['Loss'] = np.where(_df['Diff'] < 0, -_df['Diff'], 0)

    _df['Avg Gain'] = np.nan
    _df['Avg Loss'] = np.nan
    first_avg_gain = _df['Gain'][:period].mean()
    first_avg_loss = _df['Loss'][:period].mean()
    _df['Avg Gain'][period - 1] = first_avg_gain
    _df['Avg Loss'][period - 1] = first_avg_loss
    for i in range(period, _df.shape[0]):
        _df['Avg Gain'][i] = ((period - 1) * _df['Avg Gain'][i - 1] + _df['Gain'][i]) / period
        _df['Avg Loss'][i] = ((period - 1) * _df['Avg Loss'][i - 1] + _df['Loss'][i]) / period

    _df['RS'] = _df['Avg Gain'] / _df['Avg Loss']
    _df['RSI'] = 100 - 100 / (1 + _df['RS'])

    return _df


def rsi_2(df_prev: pd.DataFrame, period: int = 14, ticker_name: str = None) -> pd.DataFrame:
    if period < 1: raise ValueError('Period must be >= 1')
    if ticker_name is None:
        ticker_name = 'Close'
    _df = pd.DataFrame(df_prev[ticker_name])

    _df['Diff'] = _df[ticker_name].diff()
    _df['Gain'] = np.where(_df['Diff'] > 0, _df['Diff'], 0)
    _df['Loss'] = np.where(_df['Diff'] < 0, -_df['Diff'], 0)

    _df['Avg Gain'] = _df['Gain'].rolling(period).mean()
    _df['Avg Loss'] = _df['Loss'].rolling(period).mean()

    _df['RS'] = _df['Avg Gain'] / _df['Avg Loss']
    _df['RSI'] = 100 - 100 / (1 + _df['RS'])

    return _df


def macd(
        df_prev: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        ticker_name: str = None
) -> pd.DataFrame:
    if fast < 0: raise ValueError('Fast must be greater than 0')
    if slow < 0: raise ValueError('Slow must be greater than 0')
    if signal < 0: raise ValueError('Signal must be greater than 0')
    if fast > slow:
        temp = fast
        fast = slow
        slow = temp
    if ticker_name is None:
        ticker_name = 'Close'
    _df = pd.DataFrame(df_prev[ticker_name])

    _df[f'EMA{fast}'] = _df[ticker_name].ewm(span=fast, adjust=False).mean()
    _df[f'EMA{slow}'] = _df[ticker_name].ewm(span=slow, adjust=False).mean()
    _df[f'MACD'] = _df[f'EMA{fast}'] - _df[f'EMA{slow}']
    _df[f'Signal Line'] = _df[f'MACD'].ewm(span=signal, adjust=False).mean()

    return _df


def stochastic_oscillator(
        df_prev: pd.DataFrame,
        period: int = 14,
        ticker_name: str = None,
        type: str = 'fast'
) -> pd.DataFrame:
    if period <= 0: raise ValueError('Fast must be greater than 0')
    if ticker_name is None:
        ticker_name = 'Close'
    _df = pd.DataFrame(df_prev[ticker_name])

    """
    Assumption of the indicator: 
    closing prices should move in the same direction as the current trend
    
    - More useful in range-bound markets
    
    K: Fast 
    D: Slow
    """

    _df[f'L{period}'] = _df[ticker_name].rolling(period).min()
    _df[f'H{period}'] = _df[ticker_name].rolling(period).max()

    _df['K'] = (_df[ticker_name] - _df[f'L{period}']) / (_df[f'H{period}'] - _df[f'L{period}']) * 100
    if type != 'fast':
        _df['D'] = _df['K'].rolling(3).mean()

    return _df


def mfi(df_raw: pd.DataFrame, ticker_name: str, period: int = 14) -> pd.DataFrame:  # money flow index
    """
    :param df_raw: dataframe
    :param period: 14 as default
    :param ticker_name:
    :return: dataframe
    """

    if period <= 0: raise ValueError('Fast must be greater than 0')
    if 'Ticker' in df_raw:
        price_df = df_raw.pivot(index="Date", columns="Ticker", values="Close")
        volume_df = df_raw.pivot(index="Date", columns="Ticker", values="Volume")
        high_df = df_raw.pivot(index="Date", columns="Ticker", values="High")
        low_df = df_raw.pivot(index="Date", columns="Ticker", values="Low")
        _df = pd.DataFrame(price_df[ticker_name])
        _df['High'] = high_df[ticker_name]
        _df['Low'] = low_df[ticker_name]
        _df['Volume'] = volume_df[ticker_name]
    elif ('Close' in df_raw) and ('Volume' in df_raw) and ('High' in df_raw) and ('Low' in df_raw):
        _df = df_raw.copy()
    else:
        raise Exception('Dataframe is not in the correct format')

    _df['Typical Price'] = (_df['High'] + _df[ticker_name] + _df['Low']) / 3
    _df['+ Money Flow'] = np.where(_df['Typical Price'].diff() > 0, _df['Typical Price'] * _df['Volume'], 0)
    _df['- Money Flow'] = np.where(_df['Typical Price'].diff() < 0, -_df['Typical Price'] * _df['Volume'], 0)
    _df['Period + Money Flow'] = _df['+ Money Flow'].rolling(period).sum()
    _df['Period - Money Flow'] = _df['- Money Flow'].rolling(period).sum()

    _df['Money Flow Ratio'] = _df['Period + Money Flow'] / -_df['Period - Money Flow']
    _df['Money Flow Index'] = 100 - 100 / (1 + _df['Money Flow Ratio'])

    return _df

