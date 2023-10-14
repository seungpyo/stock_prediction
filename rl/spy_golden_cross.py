from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf

from base import *

class StockTradeAction(BaseAction):
    BUY = 0
    SELL = 1
    HOLD = 2

class SPYMAState(BaseState):
    def __init__(self, price: float, ma10: float, ma20: float, ma10_prev: float, ma20_prev: float) -> None:
        self.price: float = price
        self.ma10: float = ma10
        self.ma20: float = ma20
        self.ma10_prev: float = ma10_prev
        self.ma20_prev: float = ma20_prev
    def to_ndarray(self) -> np.ndarray:
        return np.array([self.ma10, self.ma20, self.ma10_prev, self.ma20_prev])
    def __repr__(self) -> str:
        return f"SPYMAState(ma10={self.ma10}, ma20={self.ma20}, ma10_prev={self.ma10_prev}, ma20_prev={self.ma20_prev})"
    
class SPYMAAgent(BaseAgent):
    def __init__(self) -> None:
        pass
    def select_action(self, state: SPYMAState) -> StockTradeAction:
        is_golden_cross = state.ma10_prev < state.ma20_prev and state.ma10 > state.ma20
        is_death_cross = state.ma10_prev > state.ma20_prev and state.ma10 < state.ma20
        if is_golden_cross:
            return StockTradeAction.SELL
        elif is_death_cross:
            return StockTradeAction.BUY
        else:
            return StockTradeAction.HOLD

class SPYMAEnvironment(BaseEnvironment):
    def __init__(
        self, 
        agent: SPYMAAgent, 
        ticker: str,
        start_date: date,
        end_date: date,
        initial_cash: float,
    ) -> None:
        self.ticker = ticker
        self.df = yf.download(
            tickers=ticker, 
            start=start_date.strftime("%Y-%m-%d"), 
            end=end_date.strftime("%Y-%m-%d"),
        )
        self.df.reset_index(inplace=True)
        self._current_idx = 1 # We also need to look at the previous day's MA
        self.account = Account(initial_cash)
        initial_state = SPYMAState(
            price=self.df['Close'].iloc[self._current_idx],
            ma10=self.df['Close'].rolling(10).mean().iloc[self._current_idx],
            ma20=self.df['Close'].rolling(20).mean().iloc[self._current_idx],
            ma10_prev=self.df['Close'].rolling(10).mean().iloc[self._current_idx - 1],
            ma20_prev=self.df['Close'].rolling(20).mean().iloc[self._current_idx - 1],
        )
        super().__init__(agent, initial_state)
    def reset(self) -> None:
        self._current_idx = 0
        self.account = Account(self.account.initial_cash)
        super().reset()
    def get_last_price_of_tickers(self, tickers: List[str]) -> Dict[str, float]:
        return {ticker: self.df[ticker].iloc[self._current_idx] for ticker in tickers}
    def train_on_single_episode(self) -> None:
        pass
    def perform_action(self, action: StockTradeAction) -> Tuple[SPYMAState, Optional[StockTradeAction]]:
        transaction_failed = False
        price = self.df['Close'].iloc[self._current_idx]
        if action == StockTradeAction.BUY:
            try:
                volume = self.account.예수금 // price
                # volume = int(self.account.cash // price * 0.2)
                self.account.buy(self.ticker, volume, price, self.df['Date'].iloc[self._current_idx])
            except ValueError:
                transaction_failed = True
        elif action == StockTradeAction.SELL:
            try:
                volume = self.account.종목별_보유주식수.get(self.ticker, 0)
                self.account.sell(self.ticker, volume, price, self.df['Date'].iloc[self._current_idx])
            except ValueError:
                transaction_failed = True
        else:
            pass
        next_state = SPYMAState(
            price=price,
            ma10=self.df['Close'].rolling(10).mean().iloc[self._current_idx],
            ma20=self.df['Close'].rolling(20).mean().iloc[self._current_idx],
            ma10_prev=self.df['Close'].rolling(10).mean().iloc[self._current_idx - 1],
            ma20_prev=self.df['Close'].rolling(20).mean().iloc[self._current_idx - 1],
        )
        if transaction_failed:
            action = StockTradeAction.HOLD
        self._current_idx += 1
        return next_state, action
    def evaluate_rewards(self) -> None:
        for i, snapshot in enumerate(self.snapshots):
            if i == len(self.snapshots) - 2:
                break
            next_snapshot = self.snapshots[i + 1]
            log_diff = np.log(next_snapshot.state.price) - np.log(snapshot.state.price)
            if snapshot.action == StockTradeAction.BUY:
                reward = log_diff
            elif snapshot.action == StockTradeAction.SELL:
                reward = -log_diff
            else:
                reward = 0.0
            self.snapshots[i].reward = reward
    @property
    def done(self) -> bool:
        return self._current_idx == len(self.df)


if __name__ == "__main__":
    train_environment = SPYMAEnvironment(
        agent=SPYMAAgent(),
        ticker="SPY",
        start_date=date(2010, 1, 1),
        end_date=date(2015, 1, 1),
        initial_cash=10000.0,
    )
    test_environment = SPYMAEnvironment(
        agent=SPYMAAgent(),
        ticker="SPY",
        start_date=date(2015, 1, 1),
        end_date=date(2020, 1, 1),
        initial_cash=10000.0,
    )
    train_and_test_stock_trading_agent(
        train_environment=train_environment,
        num_train_episodes=1,
        test_environment=test_environment,
    )



