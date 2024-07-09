import numpy as np
import pandas as pd


def bollinger_bands(data, period: int, step: float = 1, ticker_name: str = None) -> pd.DataFrame:
    if period < 1: raise ValueError('Period must be >= 1')
    if step <= 0: raise ValueError('Step must be > 0')
    if ticker_name is None: ticker_name = 'Close'

    dictionary = {ticker_name: data}
    _df = pd.DataFrame(data=dictionary)

    _df[f'MA{period}'] = _df[ticker_name].rolling(period).mean()
    _df['Upper Band'] = _df[f'MA{period}'] + step * _df[f'MA{period}'].rolling(period).std()
    _df['Lower Band'] = _df[f'MA{period}'] - step * _df[f'MA{period}'].rolling(period).std()

    return _df


# default period: 14
def rsi(data, period: int = 14, ticker_name: str = None) -> pd.DataFrame:
    if period < 1: raise ValueError('Period must be >= 1')
    if ticker_name is None: ticker_name = 'Close'

    """
    RSI is more useful in trending market
    """

    if isinstance(data, list):
        data = {ticker_name: data}

    _df = pd.DataFrame(data)

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


def macd(
        data,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        ticker_name: str = None
) -> pd.DataFrame:
    if fast < 0: raise ValueError('Fast must be greater than 0')
    if slow < 0: raise ValueError('Slow must be greater than 0')
    if signal < 0: raise ValueError('Signal must be greater than 0')
    if ticker_name is None: ticker_name = 'Close'
    if fast > slow:
        temp = fast
        fast = slow
        slow = temp

    if isinstance(data, list):
        data = {ticker_name: data}

    _df = pd.DataFrame(data)

    _df[f'EMA{fast}'] = _df[ticker_name].ewm(span=fast, adjust=False).mean()
    _df[f'EMA{slow}'] = _df[ticker_name].ewm(span=slow, adjust=False).mean()
    _df[f'MACD'] = _df[f'EMA{fast}'] - _df[f'EMA{slow}']
    _df[f'Signal Line'] = _df[f'MACD'].ewm(span=signal, adjust=False).mean()

    return _df


def stochastic_oscillator(
        data,
        period: int = 14,
        ticker_name: str = None,
        type: str = 'fast'
) -> pd.DataFrame:
    if period <= 0: raise ValueError('Fast must be greater than 0')
    if ticker_name is None: ticker_name = 'Close'

    """
    Assumption of the indicator: 
    closing prices should move in the same direction as the current trend
    
    - More useful in range-bound markets
    
    K: Fast 
    D: Slow
    """

    if isinstance(data, list):
        data = {ticker_name: data}

    _df = pd.DataFrame(data)

    _df[f'L{period}'] = _df[ticker_name].rolling(period).min()
    _df[f'H{period}'] = _df[ticker_name].rolling(period).max()

    _df['K'] = (_df[ticker_name] - _df[f'L{period}']) / (_df[f'H{period}'] - _df[f'L{period}']) * 100
    if type != 'fast':
        _df['D'] = _df['K'].rolling(3).mean()

    return _df


def smma(src, length: int):
    smma_values = np.zeros_like(src)
    smma_values[length - 1] = np.mean(src[:length])
    for i in range(length, len(src)):
        smma_values[i] = (smma_values[i - 1] * (length - 1) + src[i]) / length
    return smma_values


def zlema(src, length: int):
    ema1 = src.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    d = ema1 - ema2
    zlema_values = ema1 + d
    return zlema_values


def imacd_lb(df_prev: pd.DataFrame, period: int = 34, signal: int = 9):
    """
    df_prev: contains high low close
    """
    if 'High' not in df_prev or 'Low' not in df_prev or 'Close' not in df_prev:
        raise ValueError('High / Low / Close not found')

    if period <= 0: raise ValueError('Period must be greater than 0')
    if signal <= 0: raise ValueError('Period must be greater than 0')

    _df = df_prev.copy()

    _df['hlc3'] = (_df['High'] + _df['Low'] + _df['Close']) / 3
    _df['hi'] = smma(_df['High'], period)
    _df['lo'] = smma(_df['Lo2'], period)
    _df['mi'] = zlema(_df['hlc3'], period)

    def calc_md(row):
        if row['mi'] > row['hi']:
            return row['mi'] - row['hi']
        elif row['mi'] < row['lo']:
            return row['mi'] - row['lo']
        else:
            return 0

    _df['md'] = _df.apply(calc_md, axis=1)
    _df['sb'] = _df['md'].rolling(signal).mean()
    _df['sh'] = _df['md'] - _df['sb']

    return _df
