from typing import *
import numpy as np
import pandas as pd
import yfinance as yf

from dataclasses import dataclass
from enum import Enum
from datetime import datetime, date

class Order:
    def __init__(self, ticker: str, volume: int, price: float, date: date) -> None:
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
    def __init__(self, initial_cash: float):
        self.initial_cash: float = initial_cash
        self.cash: float = initial_cash
        self.orders: List[Order] = []
    def buy(self, ticker: str, volume: int, price: float, date: date) -> None:
        if volume * price > self.cash:
            raise ValueError("Not enough cash")
        self.cash -= volume * price
        self.orders.append(Order(ticker, volume, price, date))
    def sell(self, ticker: str, volume: int, price: float, date: date) -> None:
        if volume > self.num_shares.get(ticker, 0):
            raise ValueError("Not enough shares")
        self.cash += volume * price
        self.orders.append(Order(ticker, volume, price, date))
    def __repr__(self) -> str:
        return f"Account(cash={self.cash}, orders={self.orders})"
    @property
    def num_shares(self) -> Dict[str, int]:
        shares: Dict[str, int] = {}
        for order in self.orders:
            if order.ticker not in shares:
                shares[order.ticker] = 0
            shares[order.ticker] += order.volume
        return shares
    @property
    def net_worth(self) -> float:
        return self.cash + sum(order.volume * order.price for order in self.orders)
    @property
    def profit(self) -> float:
        return self.net_worth - self.initial_cash
    @property
    def num_trades(self) -> int:
        return len(self.orders)
    @property
    def elapsed_days(self) -> int:
        return (self.orders[-1].date - self.orders[0].date).days
    @property
    def cagr(self) -> float:
        return (self.net_worth / self.initial_cash) ** (365 / self.elapsed_days) - 1


