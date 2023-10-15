from datetime import date, datetime
from copy import deepcopy
import numpy as np
import pandas as pd
import yfinance as yf
from torch import nn
import torch
from base import *
from record import *

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
        self.date: date = date
        self.today_open: float = today_open
        self.ma5: float = ma5
        self.ma10: float = ma10
        self.ma20: float = ma20
        self.ma60: float = ma60
        self.ma120: float = ma120

    def to_ndarray(self) -> np.ndarray:
        arr = np.array([
            self.today_open,
            self.ma5,
            self.ma10,
            self.ma20,
            self.ma60,
            self.ma120,
        ])
        nan_indices = np.argwhere(np.isnan(arr))
        if len(nan_indices) > 0:
            raise ValueError(f"NaN detected in {self}")
        return arr

    
class DQNV1Agent(BaseAgent):
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: torch.device,
    ) -> None:
        self.model: nn.Module = model
        self.old_model: nn.Module = deepcopy(model)
        self.optimizer: torch.optim.Optimizer = optimizer
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler = lr_scheduler
        self.device = device
        self.model = self.model.to(self.device)
        self.old_model = self.old_model.to(self.device)
        self._ignore_model: bool = True

    def trust_model_from_now(self) -> None:
        self._ignore_model = False

    @torch.no_grad()
    def select_action(self, state: DQNV1State) -> StockTradeAction:
        if self._ignore_model:
            return np.random.choice(
                [
                    StockTradeAction.BUY,
                    StockTradeAction.SELL,
                    StockTradeAction.HOLD,
                ], 
                # p=[0.3, 0.3, 0.4],
            )
        try:
            input_vec = state.to_tensor().to(device=self.device).unsqueeze(0)
        except ValueError as e:
            return StockTradeAction.HOLD
        self.model.eval()
        pred = self.model(input_vec)
        return StockTradeAction(pred.argmax().to(device="cpu").item())
    
    def update_old_model(self) -> None:
        self.old_model.load_state_dict(self.model.state_dict())

class DQNTrainConfig(BaseTrainConfig):
    def __init__(
        self,
        train_episodes: int,
        batches_per_episode: int,
        batch_size: int,
        gamma: float,
        device: torch.device,
        learning_rate: float,
        lr_scheduler_step_size: int,
        lr_scheduler_gamma: float,
        trust_after_ratio: float = 0.5,
    ) -> None:
        super().__init__(train_episodes, batches_per_episode, batch_size)
        self.gamma: float = gamma
        self.device: torch.device = device
        self.learning_rate: float = learning_rate
        self.lr_scheduler_step_size: int = lr_scheduler_step_size
        self.lr_scheduler_gamma: float = lr_scheduler_gamma
        self.trust_after_ratio: float = trust_after_ratio


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
        self.train_log: List[Dict[str, float]] = []
        self.train_record: TrainRecord = TrainRecord(
            log_path=f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json",
            metadata={
                **asdict(train_config),
                "ticker": ticker,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "initial_cash": initial_cash,
            }
        )
        super().__init__(agent, initial_state)
    def get_current_state(self) -> DQNV1State:
        try:
            return DQNV1State(
                date=self.df['Date'].iloc[self._current_idx],
                today_open=self.df['Open'].iloc[self._current_idx],
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
    def train_on_single_episode(self, train_config: DQNTrainConfig, current_episode: int) -> None:
        agent: DQNV1Agent = self.agent
        if float(current_episode) / train_config.train_episodes > train_config.trust_after_ratio:
            agent.trust_model_from_now()
        if current_episode % 5 == 0:
            agent.update_old_model()
        agent.model.train()
        valid_snapshots = [snapshot for snapshot in self.snapshots if snapshot.reward is not None and not np.isnan(snapshot.state.ma120)]
        for batch_idx in range(train_config.batches_per_episode):
            random_snapshot_batch: List[ExperienceSnapshot] = np.random.choice(valid_snapshots, train_config.batch_size, replace=False,)
            state_batch: torch.Tensor = torch.stack([snapshot.state.to_tensor() for snapshot in random_snapshot_batch]).to(device=train_config.device)
            action_batch: torch.LongTensor = torch.LongTensor([snapshot.action.value for snapshot in random_snapshot_batch]).to(device=train_config.device)
            reward_batch: torch.FloatTensor = torch.FloatTensor([snapshot.reward for snapshot in random_snapshot_batch]).to(device=train_config.device)
            old_pred_q_values: torch.Tensor = agent.old_model(state_batch)
            pred_q_values: torch.Tensor = agent.model(state_batch)
            old_rewards: torch.Tensor = old_pred_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
            pred_rewards: torch.Tensor = pred_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
            target_reward_batch: torch.FloatTensor = reward_batch + train_config.gamma * old_rewards
            loss = nn.functional.mse_loss(pred_rewards, target_reward_batch)
            agent.model.zero_grad()
            loss.backward()
            agent.optimizer.step()
            agent.lr_scheduler.step()
            self.train_record.add(TrainBatchLog(
                loss=loss.item(),
                lr=agent.lr_scheduler.get_last_lr()[0],
                rewards=[r.item() for r in reward_batch],
                actions=[StockTradeAction(value=a.item()) for a in action_batch],
            ))
        self.train_record.save()
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
        last_position: Optional[StockTradeAction] = None
        last_reward: Optional[float] = None
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
                last_position = StockTradeAction.BUY
                last_reward = reward
            elif snapshot.action == StockTradeAction.SELL:
                reward = -log_diff
                last_position = StockTradeAction.SELL
                last_reward = reward
            elif snapshot.action == StockTradeAction.HOLD:
                if last_position is None:
                    reward = 0.0
                elif last_position == StockTradeAction.BUY:
                    last_reward += log_diff
                elif last_position == StockTradeAction.SELL:
                    last_reward -= log_diff
            else:
                raise ValueError(f"Unknown action: {snapshot.action}")
            self.snapshots[i].reward = reward
    @property
    def done(self) -> bool:
        return self._current_idx == len(self.df)


class DQNV1Model(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__": 
    train_config: DQNTrainConfig = DQNTrainConfig(
        train_episodes=4,
        batches_per_episode=6,
        batch_size=8,
        gamma=0.5,
        device=torch.device("cpu"),
        # device=torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"),
        learning_rate=1e-2,
        lr_scheduler_step_size=1250,
        lr_scheduler_gamma=0.25,
        trust_after_ratio=0.5,
    )
    agent_model = DQNV1Model(input_dim=6, output_dim=3)
    agent_optimizer = torch.optim.AdamW(
        params=agent_model.parameters(), 
        lr=train_config.learning_rate,
        weight_decay=1e-4,
    )
    agent_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=agent_optimizer,
        T_0=train_config.batches_per_episode * train_config.train_episodes + 1,
        T_mult=1,
        eta_min=3e-4,
    )
    agent = DQNV1Agent(agent_model, agent_optimizer, agent_lr_scheduler, train_config.device)

    ticker = "SPY"
    train_start_date = date(2010, 1, 1)
    train_end_date = date(2014, 12, 31)
    test_start_date = date(2015, 1, 1)
    test_end_date = date(2020, 1, 1)
    train_environment = DQNV1Environment(
        agent=agent,
        ticker=ticker,
        start_date=train_start_date,
        end_date=train_end_date,
        initial_cash=10000.0,
    )
    test_environment = DQNV1Environment(
        agent=agent,
        ticker=ticker,
        start_date=test_start_date,
        end_date=test_end_date,
        initial_cash=10000.0,
    )

    train_environment.train(train_config)
    test_environment.agent = train_environment.agent
    test_environment.experience_single_episode()

    def cagr_over(account: Account, start_date: date, end_date: date) -> float:
        return (account.총자산 / account.initial_cash) ** (1 / (end_date - start_date).days) - 1

    print("============ Train result ============")
    print(f"Initial cash: {train_environment.account.initial_cash}")
    print(f"예수금: {train_environment.account.예수금}")
    print(f"보유주식 평가액: {train_environment.account.보유주식_평가액}")
    print(f"총자산: {train_environment.account.총자산}")
    print(f"수익률 (%): {train_environment.account.수익 / train_environment.account.initial_cash * 100}%")
    print(f"실현 수익률 (%): {train_environment.account.실현수익 / train_environment.account.initial_cash * 100}%")
    print(f"CAGR (%): {cagr_over(train_environment.account, train_start_date, train_end_date)* 100}%")
    print(f"num_trades: {train_environment.account.num_trades}")

    print("============ Test result ============")
    print(f"Initial cash: {train_environment.account.initial_cash}")
    print(f"예수금: {test_environment.account.예수금}")
    print(f"보유주식 평가액: {test_environment.account.보유주식_평가액}")
    print(f"총자산: {test_environment.account.총자산}")
    print(f"수익률 (%): {test_environment.account.수익 / train_environment.account.initial_cash * 100}%")
    print(f"실현 수익률 (%): {test_environment.account.실현수익 / train_environment.account.initial_cash * 100}%")
    print(f"CAGR (%): {cagr_over(test_environment.account, test_start_date, test_end_date)* 100}%")
    print(f"num_trades: {test_environment.account.num_trades}")

    print(f"Train record is saved at {train_environment.train_record.log_path}")




