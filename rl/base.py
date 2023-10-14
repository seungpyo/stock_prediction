from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Enum
from typing import *

import numpy as np
import torch

from order import Account

class BaseAction(Enum):
    pass

class BaseState(ABC):
    @abstractmethod
    def to_ndarray(self) -> np.ndarray:
        raise NotImplementedError
    def to_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.to_ndarray())

class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state: BaseState) -> BaseAction:
        raise NotImplementedError
    
class ExperienceSnapshot:
    def __init__(self, state: BaseState, action: BaseAction, reward: Optional[float] = None) -> None:
        self.state: BaseState = state
        self.action: BaseAction = action
        self.reward: Optional[float] = reward
    
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
    def train(self, num_episodes: int) -> None:
        for _ in range(num_episodes):
            self.experience_single_episode()
            self.train_on_single_episode()
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
    def train_on_single_episode(self) -> None:
        raise NotImplementedError
    @abstractmethod
    def perform_action(self, action: BaseAction) -> Tuple[BaseState, Optional[BaseAction]]:
        raise NotImplementedError
    @abstractmethod
    def evaluate_rewards(self) -> None:
        raise NotImplementedError
    @property
    def done(self) -> bool:
        raise NotImplementedError


def train_and_test_stock_trading_agent(
        train_environment: BaseEnvironment,
        num_train_episodes: int,
        test_environment: BaseEnvironment,
    ) -> None:
    train_environment.train(num_train_episodes)
    test_environment.agent = train_environment.agent
    test_environment.experience_single_episode()
    print("Test CAGR(%):", test_environment.account.cagr * 100)
    print("Test profit(%):", test_environment.account.profit * 100)
    print("Test number of trades:", test_environment.account.num_trades)
    print("Test elapsed days:", test_environment.account.elapsed_days)
    print("Test net worth:", test_environment.account.net_worth)
    print("Test final balance:", test_environment.account.balance)

