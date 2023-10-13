from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, date
from typing import *

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rl.order import Account

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
        self._snapshots: List[ExperienceSnapshot] = []
    def reset(self) -> None:
        self._current_state = self.initial_state
        self._snapshots = []
    def experience_single_episode(self) -> None:
        self.reset()
        while not self.done:
            action = self.agent.select_action(self._current_state)
            next_state = self.perform_action(action)
            self._snapshots.append(ExperienceSnapshot(self._current_state, action))
            self._current_state = next_state
    @abstractmethod
    def perform_action(self, action: BaseAction) -> BaseState:
        raise NotImplementedError
    @abstractmethod
    def evaluate_rewards(self) -> None:
        raise NotImplementedError
    @abstractmethod
    @property
    def done(self) -> bool:
        raise NotImplementedError
