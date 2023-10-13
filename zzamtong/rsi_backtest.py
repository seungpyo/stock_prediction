from backtesting import Backtest, Strategy

import pandas as pd
import numpy as np
import os


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



def get_snp500(filename: str = "SP500.csv") -> pd.DataFrame:
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), filename),
        index_col=0, 
        parse_dates=True, 
        infer_datetime_format=True,
    )
    s = df["SP500"].replace(".", method="ffill").astype(np.float32)
    df["Open"] = s
    df["High"] = s
    df["Low"] = s
    df["Close"] = s
    df.drop(columns=["SP500"], inplace=True)
    return df

class RSITxn(Strategy):
    rsi_lower = 40
    rsi_upper = 60
    window_size = 30
    def init(self):
        close = self.data.Close.to_series()
        self.rsi = self.I(get_rsi_of_day, close, self.window_size)

    def next(self):
        # if self.rsi[-1] < self.rsi_lower:
        if self.rsi[-2] <= self.rsi_lower and self.rsi[-1] > self.rsi_lower:
            self.buy()
            # self.sell()
        # elif self.rsi[-1] > self.rsi_upper:
        elif self.rsi[-2] >= self.rsi_upper and  self.rsi[-1] < self.rsi_upper:
            # self.buy()
            self.sell()


SNP500 = get_snp500()

bt = Backtest(
    data=SNP500, 
    strategy=RSITxn,
    cash=1000000, 
    # commission=.002,
    trade_on_close=True,
    exclusive_orders=True,
    # exclusive_orders=False,
)

output = bt.run()
print(output)

bt.plot(results=output)