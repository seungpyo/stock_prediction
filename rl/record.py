from dataclasses import dataclass, asdict
from datetime import date
from typing import *
import json
from matplotlib import pyplot as plt
import numpy as np

from base import BaseTrainConfig, StockTradeAction

@dataclass(frozen=True)
class TrainBatchLog:
    loss: float
    lr: float
    rewards: List[float]
    actions: List[StockTradeAction]
    
class TrainRecord:
    def __init__(self, log_path: str, metadata: Dict[str, Any]) -> None:
        self.log_path = log_path
        self._logs: List[TrainBatchLog] = []
        self.metadata: Dict[str, Any] = metadata

    def add(self, log: TrainBatchLog) -> None:
        self._logs.append(log)
    
    def save(self) -> None:
        save_dict: Dict[str, Any] = {
            "enum_map": {
                "StockTradeAction": {
                    "BUY": StockTradeAction.BUY.value,
                    "SELL": StockTradeAction.SELL.value,
                    "HOLD": StockTradeAction.HOLD.value,
                },
            },
            "metadata": self.metadata,
            "batch_logs": [asdict(l) for l in self._logs],
        }
        for k, v in save_dict["metadata"].items():
            if isinstance(v, date):
                save_dict["metadata"][k] = v.strftime("%Y-%m-%d")
        for i, l in enumerate(save_dict["batch_logs"]):
            save_dict["batch_logs"][i]["actions"] = [a.value for a in l["actions"]]
        with open(self.log_path, "+w") as f:
            json.dump(save_dict, f, ensure_ascii=False, indent=2)

    def plot(self) -> None:
        plt.subplot(2, 2, 1)
        plt.title("Loss per batch")
        plt.yscale("log")
        plt.plot([log.loss for log in self._logs])

        plt.subplot(2, 2, 2)
        plt.title("Average reward per batch")
        plt.plot([np.mean(log.rewards) for log in self._logs])

        plt.subplot(2, 2, 3)
        plt.title("Number of actions in a batch")
        plt.plot([len([a for a in log.actions if a == StockTradeAction.BUY]) for log in self._logs], label="num_buy")
        plt.plot([len([a for a in log.actions if a == StockTradeAction.HOLD]) for log in self._logs], label="num_hold")
        plt.plot([len([a for a in log.actions if a == StockTradeAction.SELL]) for log in self._logs], label="num_sell")
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.title("Learning rate per batch")
        plt.plot([log.lr for log in self._logs])

        plt.show()

    @staticmethod
    def load(
        log_path: str,
    ) -> 'TrainRecord':
        save_dict: Dict[str, Any] = {}
        with open(log_path, "r") as f:
            save_dict = json.load(f)
        metadata: Dict[str, Any] = save_dict["metadata"]
        logs: List[TrainBatchLog] = [
            TrainBatchLog(
                loss=l["loss"],
                lr=l["lr"],
                rewards=l["rewards"],
                actions=[StockTradeAction(value=a) for a in l["actions"]],
            )
            for l in save_dict["batch_logs"]
        ]
        logger = TrainRecord(
            log_path=log_path,
            metadata=metadata,
        )
        for l in logs:
            logger.add(l)
        return logger
        
@dataclass(frozen=True)
class TradeLog:
    date: date
    ticker: str
    action: Optional[StockTradeAction]
    open_price: float
    close_price: float
    high_price: float
    low_price: float
    volume: float
    reward: Optional[float]
    balance: float

class TradeRecord:
    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        self._logs: List[TradeLog] = []

    def add(self, log: TradeLog) -> None:
        self._logs.append(log)

    def save(self) -> None:
        save_dict: Dict[str, Any] = {
            "trade_logs": [asdict(l) for l in self._logs],
        }
        with open(self.log_path, "+w") as f:
            json.dump(save_dict, f, ensure_ascii=False, indent=2)
        
    @staticmethod
    def load(
        log_path: str,
    ) -> 'TradeRecord':
        save_dict: Dict[str, Any] = {}
        with open(log_path, "r") as f:
            json.load(save_dict, f)
        logs: List[TradeLog] = [
            TradeLog(
                date=l["date"],
                ticker=l["ticker"],
                action=StockTradeAction(value=l["action"]) if l["action"] is not None else None,
                open_price=l["open_price"],
                close_price=l["close_price"],
                high_price=l["high_price"],
                low_price=l["low_price"],
                volume=l["volume"],
                reward=l["reward"],
                balance=l["balance"],
            )
            for l in save_dict["trade_logs"]
        ]
        logger = TradeRecord(
            log_path=log_path,
        )
        for l in logs:
            logger.add(l)
        return logger
    
    def plot(self) -> None:
        plt.subplot(2, 2, 1)
        plt.title("Balance per day")
        plt.plot([l.balance for l in self._logs])

        plt.subplot(2, 2, 2)
        plt.title("Reward per day")
        plt.plot([l.reward for l in self._logs])

        plt.subplot(2, 2, 3)
        plt.title("Closing price per day")
        plt.plot([l.close_price for l in self._logs])

        plt.show()
