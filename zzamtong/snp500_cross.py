from backtesting import Backtest, Strategy
from backtesting.test import SMA, GOOG
from backtesting.lib import crossover
import pandas as pd
import numpy as np
import os
import yfinance as yf



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

class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()

SNP500 = get_snp500()

bt = Backtest(
    # data=SNP500, 
    data=GOOG, 
    strategy=SmaCross,
    cash=1000000, 
    # commission=.002,
    trade_on_close=True,
    exclusive_orders=True,
    # exclusive_orders=False,
)

output = bt.run()
print(output)

bt.plot(results=output)