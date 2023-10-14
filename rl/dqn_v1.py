from datetime import date
from copy import deepcopy
import numpy as np
import pandas as pd
import yfinance as yf
from torch import nn
import torch


from base import *

class StockTradeAction(BaseAction):
    BUY = 0
    SELL = 1
    HOLD = 2

class DQNV1State(BaseState):
    def __init__(
        self, 
        date: date,
        today_open: float, 
        ma5: float, 
        ma10: float, 
        ma20: float, 
        ma60: float, 
        ma120: float,
    ) -> None:
        self.date: date = date,
        self.today_open: float = today_open
        self.ma5: float = ma5
        self.ma10: float = ma10
        self.ma20: float = ma20
        self.ma60: float = ma60
        self.ma120: float = ma120

    def to_ndarray(self) -> np.ndarray:
        return np.array([
            self.today_open,
            self.ma5,
            self.ma10,
            self.ma20,
            self.ma60,
            self.ma120,
        ])
    
class DQNV1Agent(BaseAgent):
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler.LRScheduler) -> None:
        self.model: nn.Module = model
        self.old_model: nn.Module = deepcopy(model)
        self.optimizer: torch.optim.Optimizer = optimizer
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler = lr_scheduler

    @torch.no_grad()
    def select_action(self, state: DQNV1State) -> StockTradeAction:
        input_vec = state.to_tensor()
        pred = self.model(input_vec)
        return StockTradeAction(pred.argmax().item())
    
    def update_old_model(self) -> None:
        self.old_model.load_state_dict(self.model.state_dict())

class DQNV1Environment(BaseEnvironment):
    def __init__(
        self, 
        agent: DQNV1Agent, 
        ticker: str,
        start_date: date,
        end_date: date,
        initial_cash: float,
    ) -> None:
        self.ticker = ticker
        self.df: pd.DataFrame = yf.download(
            tickers=ticker, 
            start=start_date.strftime("%Y-%m-%d"), 
            end=end_date.strftime("%Y-%m-%d"),
        )
        self.df.reset_index(inplace=True)
        self._current_idx = 120 - 1
        assert self._current_idx < len(self.df), "Whole dataframe is less than MA120"
        self.account = Account(initial_cash, self.get_last_price_of_tickers)
        initial_state = self.get_current_state()
        super().__init__(agent, initial_state)
    def get_current_state(self) -> DQNV1State:
        try:
            return DQNV1State(
                date=self.df['Date'].iloc[self._current_idx],
                today_open=self.df['Open'].iloc[self._current_index],
                ma5=self.df['Close'].rolling(5).mean().iloc[self._current_idx],
                ma10=self.df['Close'].rolling(10).mean().iloc[self._current_idx],
                ma20=self.df['Close'].rolling(20).mean().iloc[self._current_idx],
                ma60=self.df['Close'].rolling(60).mean().iloc[self._current_idx],
                ma120=self.df['Close'].rolling(120).mean().iloc[self._current_idx],
            )
        except IndexError as e:
            raise e

    def reset(self) -> None:
        self._current_idx = 0
        self.account = Account(self.account.initial_cash, self.get_last_price_of_tickers)
        super().reset()
    def get_last_price_of_tickers(self, tickers: List[str]) -> Dict[str, float]:
        return {ticker: self.df.iloc[-1]["Close"] for ticker in tickers}
    def train_on_single_episode(self) -> None:
        agent: DQNV1Agent = self.agent
        agent.model.train()
        num_batches: int = 100
        batch_size = 64
        gamma = 0.2
        for batch_idx in range(num_batches):
            random_snapshot_batch: List[ExperienceSnapshot] = np.random.choice(self.snapshots, batch_size, replace=False,)
            state_batch: torch.Tensor = torch.stack([snapshot.state.to_tensor() for snapshot in random_snapshot_batch])
            action_batch: torch.LongTensor = torch.LongTensor([snapshot.action] for snapshot in random_snapshot_batch)
            reward_batch: torch.FloatTensor = torch.FloatTensor([snapshot.reward] for snapshot in random_snapshot_batch)
            old_pred_q_values: torch.Tensor = agent.old_model(state_batch)
            pred_q_values: torch.Tensor = agent.model(state_batch)
            old_rewards: torch.Tensor = old_pred_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
            pred_rewards: torch.Tensor = pred_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
            target_reward_batch: torch.FloatTensor = reward_batch * gamma * old_rewards
            loss = nn.functional.mse_loss(pred_rewards, target_reward_batch)
            agent.model.zero_grad()
            loss.backward()
            agent.optimizer.step()
            agent.lr_scheduler.step()
    def perform_action(self, action: StockTradeAction) -> Tuple[Optional[DQNV1State], Optional[StockTradeAction]]:
        transaction_failed = False
        today_close = self.df['Close'].iloc[self._current_idx]
        if action == StockTradeAction.BUY:
            try:
                volume = self.account.예수금 // today_close
                self.account.buy(self.ticker, volume, today_close, self.df['Date'].iloc[self._current_idx])
            except ValueError:
                transaction_failed = True
        elif action == StockTradeAction.SELL:
            try:
                volume = self.account.종목별_보유주식수.get(self.ticker, 0)
                self.account.sell(self.ticker, volume, today_close, self.df['Date'].iloc[self._current_idx])
            except ValueError:
                transaction_failed = True
        else:
            pass
        self._current_idx += 1
        try:
            next_state = self.get_current_state()
        except IndexError as e:
            next_state = None
        if transaction_failed:
            action = StockTradeAction.HOLD
        return next_state, action
    def evaluate_rewards(self) -> None:
        for i, snapshot in enumerate(self.snapshots):
            if i == len(self.snapshots) - 2:
                break
            next_snapshot = self.snapshots[i + 1]
            today_state: DQNV1State = snapshot.state
            tomrrow_state: DQNV1State = next_snapshot.state
            today_price = self.df.loc[self.df["Date"] == today_state.date]["Close"].iloc[0]
            tomorrow_price = self.df.loc[self.df["Date"] == tomrrow_state.date]["Close"].iloc[0]
            log_diff = np.log(tomorrow_price / today_price)
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
    train_environment = DQNV1Environment(
        agent=DQNV1Agent(),
        ticker="SPY",
        start_date=date(2010, 1, 1),
        end_date=date(2015, 1, 1),
        initial_cash=10000.0,
    )
    test_environment = DQNV1Environment(
        agent=DQNV1Agent(),
        ticker="SPY",
        start_date=date(2015, 1, 1),
        end_date=date(2020, 1, 1),
        initial_cash=10000.0,
    )
    train_and_test_stock_trading_agent(
        train_environment=train_environment,
        num_train_episodes=10,
        test_environment=test_environment,
    )



