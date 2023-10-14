from typing import *
import numpy as np
import pandas as pd
import yfinance as yf

from dataclasses import dataclass
from enum import Enum
from datetime import datetime, date

class Order:
    def __init__(self, ticker: str, volume: int, price: float, date: date) -> None:
        if volume == 0:
            raise ValueError("volume must be non-zero when calling Order.__init__()")
        self.ticker: str = ticker
        self.volume: int = volume
        self.price: float = price
        self.date: date = date
    def __repr__(self) -> str:
        return f"Order(ticker={self.ticker}, volume={self.volume}, price={self.price}, date={self.date})"
    @property
    def is_long(self) -> bool:
        return self.volume > 0
    @property
    def is_short(self) -> bool:
        return self.volume < 0
    
class Account:
    def __init__(self, initial_cash: float, last_price_fn: Callable[[List[str]], Dict[str, float]]):
        self.initial_cash: float = initial_cash
        self.orders: List[Order] = []
        self.last_price_fn = last_price_fn
    def buy(self, ticker: str, volume: int, price: float, date: date) -> None:
        if volume * price > self.예수금:
            raise ValueError("Not enough cash")
        if volume <= 0:
            raise ValueError("Volume must be positive when calling Account.buy()")
        self.orders.append(Order(ticker, volume, price, date))
    def sell(self, ticker: str, volume: int, price: float, date: date) -> None:
        if volume > self.종목별_보유주식수.get(ticker, 0):
            raise ValueError("Not enough shares")
        if volume <= 0:
            raise ValueError("Volume must be positive when calling Account.sell()")
        self.orders.append(Order(ticker, -volume, price, date))
    def __repr__(self) -> str:
        return f"Account(cash={self.예수금}, orders={self.orders})"
    @property
    def 종목별_보유주식수(self) -> Dict[str, int]:
        shares: Dict[str, int] = {}
        for order in self.orders:
            if order.ticker not in shares:
                shares[order.ticker] = 0
            shares[order.ticker] += order.volume
        return shares
    @property
    def 보유주식_평가액(self) -> float:
        # return sum(order.volume * order.price for order in self.orders if order.is_long)
        holding_tickers = [ticker for ticker, shares in self.종목별_보유주식수.items() if shares > 0]
        last_prices = self.last_price_fn(holding_tickers)
        return sum(shares * last_prices[ticker] for ticker, shares in self.종목별_보유주식수.items() if shares > 0)
    @property
    def 예수금(self) -> float:
        return self.initial_cash - sum(order.price * order.volume for order in self.orders)
    @property
    def 총자산(self) -> float:
        return self.예수금 + self.보유주식_평가액
    @property
    def 수익(self) -> float:
        return self.총자산 - self.initial_cash
    @property
    def 실현수익(self) -> float:
        return self.예수금 - self.initial_cash
    @property
    def num_trades(self) -> int:
        return len(self.orders)
    @property
    def elapsed_days(self) -> int:
        return (self.orders[-1].date - self.orders[0].date).days
    @property
    def cagr(self) -> float:
        return (self.총자산 / self.initial_cash) ** (365 / self.elapsed_days) - 1


