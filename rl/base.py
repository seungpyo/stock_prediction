from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Enum
from typing import *
from tqdm import tqdm

import numpy as np
import torch

from order import Account

class BaseAction(Enum):
    pass

class StockTradeAction(BaseAction):
    BUY = 0
    SELL = 1
    HOLD = 2

class BaseState(ABC):
    @abstractmethod
    def to_ndarray(self) -> np.ndarray:
        raise NotImplementedError
    def to_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.to_ndarray()).to(dtype=torch.float32)

class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state: BaseState) -> BaseAction:
        raise NotImplementedError
    
class ExperienceSnapshot:
    def __init__(self, state: BaseState, action: BaseAction, reward: Optional[float] = None) -> None:
        self.state: BaseState = state
        self.action: BaseAction = action
        self.reward: Optional[float] = reward
    
@dataclass
class BaseTrainConfig:
    train_episodes: int
    batches_per_episode: int
    batch_size: int

class BaseEnvironment:
    def __init__(self, agent: BaseAgent, initial_state: BaseState) -> None:
        self.agent: BaseAgent = agent
        self.initial_state: BaseState = initial_state
        self._current_state: BaseState = initial_state
        self.snapshots: List[ExperienceSnapshot] = []
    @property
    def current_state(self) -> BaseState:
        return self._current_state
    def reset(self) -> None:
        self._current_state = self.initial_state
        self.snapshots = []
    def train(self, train_config: BaseTrainConfig) -> None:
        print("Training with config:")
        print(train_config)
        for current_episode in tqdm(range(train_config.train_episodes)):
            self.experience_single_episode()
            self.train_on_single_episode(train_config=train_config, current_episode=current_episode)
    def test(self) -> None:
        self.experience_single_episode()
    def experience_single_episode(self) -> None:
        self.reset()
        while not self.done:
            action = self.agent.select_action(self._current_state)
            next_state, alternative_action = self.perform_action(action)
            if alternative_action is not None:
                action = alternative_action
            self.snapshots.append(ExperienceSnapshot(self._current_state, action))
            self._current_state = next_state
        self.evaluate_rewards()
    @abstractmethod
    def train_on_single_episode(self, train_config: BaseTrainConfig, current_episode: int) -> None:
        raise NotImplementedError
    @abstractmethod
    def perform_action(self, action: BaseAction) -> Tuple[Optional[BaseState], Optional[BaseAction]]:
        raise NotImplementedError
    @abstractmethod
    def evaluate_rewards(self) -> None:
        raise NotImplementedError
    @property
    def done(self) -> bool:
        raise NotImplementedError


def train_and_test_stock_trading_agent(
        train_environment: BaseEnvironment,
        train_config: BaseTrainConfig,
        test_environment: BaseEnvironment,
    ) -> None:
    train_environment.train(train_config)
    test_environment.agent = train_environment.agent
    test_environment.experience_single_episode()

    print(f"Initial cash: {train_environment.account.initial_cash}")
    print(f"예수금: {test_environment.account.예수금}")
    print(f"보유주식 평가액: {test_environment.account.보유주식_평가액}")
    print(f"총자산: {test_environment.account.총자산}")
    print(f"수익률 (%): {test_environment.account.수익 / train_environment.account.initial_cash * 100}%")
    print(f"실현 수익률 (%): {test_environment.account.실현수익 / train_environment.account.initial_cash * 100}%")
    print(f"CAGR (%): {test_environment.account.cagr * 100}%")
    print(f"num_trades: {test_environment.account.num_trades}")

