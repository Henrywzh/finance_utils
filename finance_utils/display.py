from matplotlib import pyplot as plt

from .indicators import *


def plot(data, indicator):
    pass


def plot_macd(df_prev: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, ticker_name: str = None):
    _df = macd(df_prev, fast, slow, signal, ticker_name=ticker_name)

    plt.figure(figsize=(12, 6))
    x = pd.to_datetime(_df.index.values)

    plt.subplot(2, 1, 1)
    plt.plot(x, _df[ticker_name], label=ticker_name)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x, _df['MACD'], label='MACD')
    plt.plot(x, _df['Signal Line'], label='Signal Line')
    plt.bar(x, _df['MACD'] - _df['Signal Line'], label='MACD - Signal')
    plt.legend()

    # plt.savefig(f'{ticker_name}_MACD.pdf')
    plt.show()


def plot_oscillator(df_prev: pd.DataFrame, period: int = 14, type: str = 'fast', ticker_name: str = None):
    _df = stochastic_oscillator(df_prev, period, type=type, ticker_name=ticker_name)

    plt.figure(figsize=(12, 6))
    x = pd.to_datetime(_df.index.values)

    plt.subplot(2, 1, 1)
    plt.plot(x, _df[ticker_name], label=ticker_name)
    plt.legend()

    plt.subplot(2, 1, 2)
    if type == 'fast':
        plt.plot(x, _df['K'], label=f'%Fast Oscillator: {period}')
    else:
        plt.plot(x, _df['D'], label=f'%Slow Oscillator (%K MA3): {period}')
    plt.legend()
    plt.show()


#
# class Display:
#     def __init__(self, _df: pd.DataFrame, name: str):
#         """
#         :param _df: a dataframe
#         :param name: the name of the indicator
#         """
#         self.df = _df
#         self.name = name
#
#     def safe_to_pdf(self, source=None):
#         if source is None:
#             source = ''
#         plt.savefig(f'{source}/{self.name}.pdf')
#
#     def __repr__(self):
#         return f'Name: {self.name}\nDataframe: {self.df.info()}'
#
