# finance_utils
Tools for backtesting, portfolio management, and more...

## backtest.py
class: Backtest
Info: Given a data frame with columns 'Price' (Benchmark), 'Value' (Strategy), runs backtest automatically.
Example: 
```
from finance_utils.backtest import *
df = yf.download(tickers=['SPY', 'MSFT'], start='2000-01-01', end='2024-09-01')
msft = pd.DataFrame()
msft['Price'] = df[('Adj Close', 'SPY')]
msft['Value'] = df[('Adj Close', 'MSFT')]

test_msft = Backtest(msft)
```

## portfolio.py
class: Portfolio
Info: Rebalance the portfolio according to the weights, and calculate the portfolio values and other related statistical data.
Example: 
```
from finance_utils.portfolio import *
df = yf.download(tickers=['AAPL', 'MSFT', 'AMZN', 'NVDA'], start='2015-01-01', end='2024-09-11')
initial_capital = 1_000_000
my_portfolio = Portfolio(df['Adj Close'], initial_capital, benchmark='SPY')
my_portfolio.run()
my_portfolio.plot()
```

## Installation
```
pip install git+https://github.com/Henrywzh/finance_utils.git#egg=finance_utils
```
