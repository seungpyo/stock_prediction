from typing import *
from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
import backtesting as bt

from dataclasses import dataclass
from enum import Enum
from datetime import date

from rl.order import Account

class StockTradeAction(Enum):
    BUY = 0
    SELL = 1
    HOLD = 2

class BaseState:
    def to_ndarray(self) -> np.ndarray:
        raise NotImplementedError
    def to_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.to_ndarray())
 
@dataclass(frozen=True)
class DQNMemoryItem:
    state: BaseState
    action: StockTradeAction
    reward: float
    next_state: BaseState
    done: bool

class DQNMemory:
    def __init__(self):
        self._buf: List[DQNMemoryItem] = []

    def remember(self, item: DQNMemoryItem):
        self._buf.append(item)

    def random_sample(self, batch_size: int) -> List[DQNMemoryItem]:
        return np.random.choice(self._buf, batch_size, replace=False)
    
    def __len__(self) -> int:
        return len(self._buf)
    
    def reset(self):
        self._buf = []

class StockTradeNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(StockTradeNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class StockTradeAgent:
    def __init__(self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
    ):
        self.model: nn.Module = model
        self.old_model: nn.Module = model
        self.optimizer: torch.optim.Optimizer = optimizer
        self.lr_scheduler: torch.optim.lr_scheduler._LRScheduler = lr_scheduler
        self.device: torch.device = device
        self.model = self.model.to(self.device)
        self.old_model = self.old_model.to(self.device)

    @torch.no_grad()
    def select_action(self, state: BaseState, use_random_action: bool) -> StockTradeAction:
        if use_random_action:
            action_index = np.random.choice(len(StockTradeAction))
        else:
            action_index = self.model(state.to_tensor().to(device=self.device)).argmax().item()
        return StockTradeAction(action_index)

    def train_step(self, batch: List[DQNMemoryItem], gamma: float = 0.9) -> Dict[str, torch.Tensor]:
        state_batch = torch.stack([item.state.to_tensor().to(device=self.device) for item in batch])
        action_batch = torch.tensor([item.action.value for item in batch], device=self.device)
        reward_batch = torch.tensor([item.reward for item in batch], device=self.device)
        old_q_values = self.old_model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        target_reward_batch = reward_batch + gamma * old_q_values
        q_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.mse_loss(q_values, target_reward_batch)
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        return {
            "loss": loss,
            "q_values": q_values,
            "target_reward_batch": target_reward_batch,
            "old_q_values": old_q_values,
        }

    def update_old_model(self) -> None:
        self.old_model.load_state_dict(self.model.state_dict())


class StockTradeState(BaseState):
    def __init__(self, sub_df: pd.DataFrame, account: Account) -> None:
        self.account = account
        self.df = sub_df
    @property
    def start_date(self) -> date:
        return self.df["date"].iloc[0]
    @property
    def end_date(self) -> date:
        return self.df["date"].iloc[-1]
    def to_ndarray(self) -> np.ndarray:
        lookback_array = np.stack([
            self.df['Open'].values,
            self.df['High'].values,
            self.df['Low'].values,
            self.df['Close'].values,
            self.df['Volume'].values,
        ])
        if lookback_array.shape[1] < self.days_to_lookback:
            raise ValueError(f"lookback_array.shape[1] < self.days_to_lookback: {lookback_array.shape[1]} < {self.days_to_lookback}")
        lookback_array = lookback_array.reshape(-1)
        arr = np.concatenate([lookback_array, np.array([self.account.cash / self.account.initial_cash])])
        return lookback_array


class StockTradingEmulator:
    def __init__(self, ticker: str, start_date: date, end_date: date):
        self.ticker = ticker
        self.start_date = start_date.strftime("%Y-%m-%d")
        self.end_date = end_date.strftime("%Y-%m-%d")
        self.df = yf.download(
            tickers=self.ticker, 
            start=self.start_date, 
            end=self.end_date,
        )
        self.df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)

    def __call__(self, state: StockTradeState, action: StockTradeAction) -> Tuple[float, BaseState, bool]:
        raise NotImplementedError
    
class SimpleStockTradingEmulator(StockTradingEmulator):
    def __call__(self, state: StockTradeState, action: StockTradeAction) -> Tuple[float, BaseState, bool]:
        done = state.end_date == self.end_date
        match action:
            case StockTradeAction.BUY:
                state.account.buy(self.ticker, state.df['Close'].iloc[-1])
                pass
            case StockTradeAction.SELL:
                pass
            case StockTradeAction.HOLD:
                return 0.0, state, False
        return reward, next_state, done


class StockTradingEnvironment:
    def __init__(
        self,
        agent: StockTradeAgent,
        memory: DQNMemory,
        initial_cash: float,
        commission_rate: float,
        action_emulator: Callable[[BaseState, StockTradeAction], Tuple[float, BaseState, bool]],
        initial_state: BaseState,
    ) -> None:
        self.agent = agent
        self.memory = memory
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.action_function = action_emulator
        self.current_state = initial_state
        self._initial_state = initial_state

    def reset(self) -> BaseState:
        self.cash = self.initial_cash
        self.stock = 0
        self.memory.reset()
        self.current_state = self._initial_state
    
    def experience_single_step(self, use_random_action: bool) -> DQNMemoryItem:
        action = self.agent.select_action(self.current_state, use_random_action=use_random_action)
        reward, next_state, done = self.action_function(self.current_state, action)
        return DQNMemoryItem(self.current_state, action, reward, next_state, done)
    
    def experience_single_episode(self) -> None:
        self.reset()
        done = False
        epsilon = 0.01
        epsilon_decay = 0.99
        while not done:
            use_random_action = np.random.random() < epsilon
            epsilon *= epsilon_decay
            memory_item = self.experience_single_step(use_random_action=use_random_action)
            self.memory.remember(memory_item)
            self.current_state = memory_item.next_state
            done = memory_item.done

    def train_single_episode(self, batch_size: int) -> None:
        batch = self.memory.random_sample(batch_size)
        self.agent.train_step(batch)
        self.agent.update_old_model()

    def train_agent(self, batch_size: int, num_episodes: int) -> None:
        for _ in range(num_episodes):
            self.experience_single_episode()
            self.train_single_episode(batch_size)
