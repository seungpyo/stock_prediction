import yfinance as yf
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.test import SMA, GOOG
from backtesting.lib import crossover
from random import random


ticker = "SPY"
start = "2012-01-01"
end = "2021-01-01"
result_filename = f"{ticker}_{start}_{end}.html"

spy = yf.download(
    ticker,
    start=start,
    end=end,
)


def get_rsi_of_day(arr: pd.Series, window_days: int) -> pd.Series:
    """
    Returns `n`-period simple moving average of array `arr`.
    """
    delta = arr.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.rolling(window_days).mean()
    roll_down = down.abs().rolling(window_days).mean()
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


class RSITxn(Strategy):
    rsi_lower = 40
    rsi_upper = 60
    window_size = 30
    def init(self):
        close = self.data.Close.to_series()
        self.rsi = self.I(get_rsi_of_day, close, self.window_size)
        self.buy()

    def next(self):
        # if self.rsi[-1] < self.rsi_lower:
        if self.rsi[-2] <= self.rsi_lower and self.rsi[-1] > self.rsi_lower:
            self.buy()
            # self.sell()
        # elif self.rsi[-1] > self.rsi_upper:
        elif self.rsi[-2] >= self.rsi_upper and  self.rsi[-1] < self.rsi_upper:
            # self.buy()
            self.sell()


class Monkey(Strategy):
    def init(self):
        pass
    def next(self):
        if random() > .5:
            self.buy()
        else:
            self.sell()

class SmaCross(Strategy):
    n1 = 3
    n2 = 10

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()


bt = Backtest(
    data=spy, 
    strategy=SmaCross,
    # strategy=RSITxn,
    cash=1000000, 
    # commission=.002,
    trade_on_close=True,
    exclusive_orders=True,
    # exclusive_orders=False,
)

output = bt.run()
print(output)

bt.plot(
    filename=result_filename,
    results=output,
)