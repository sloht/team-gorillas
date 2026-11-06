"""
rl_common.py — shared include file for a game-playing reinforcement learning project.

Import from your training, evaluation and play scripts:

    from rl_common import (
        Config, set_seed, get_device, make_env, ReplayBuffer,
        LinearSchedule, Logger, save_checkpoint, load_checkpoint, evaluate,
    )

Dependencies: numpy (required), torch and gymnasium (optional but expected).
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # torch is optional for non-neural agents
    torch = None  # type: ignore

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    """All hyperparameters in one place. Override fields or load from JSON."""

    # Environment
    env_id: str = "CartPole-v1"
    seed: int = 42
    max_episode_steps: Optional[int] = None
    render_mode: Optional[str] = None  # "human", "rgb_array", or None

    # Training
    total_steps: int = 200_000
    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 100_000
    learning_starts: int = 1_000
    train_every: int = 1
    target_update_every: int = 1_000
    grad_clip: Optional[float] = 10.0

    # Exploration (epsilon-greedy)
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000

    # Network
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])

    # Evaluation / logging / checkpoints
    eval_every: int = 10_000
    eval_episodes: int = 10
    log_every: int = 1_000
    run_dir: str = "runs"
    run_name: Optional[str] = None
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    def __post_init__(self) -> None:
        if self.run_name is None:
            self.run_name = f"{self.env_id}_{time.strftime('%Y%m%d-%H%M%S')}"

    @property
    def out_dir(self) -> Path:
        return Path(self.run_dir) / str(self.run_name)

    def save(self, path: Optional[os.PathLike] = None) -> Path:
        path = Path(path) if path else self.out_dir / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path: os.PathLike) -> "Config":
        return cls(**json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------- #
# Reproducibility & hardware
# --------------------------------------------------------------------------- #

def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if present) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto"):
    """Return the best available torch device (or the one requested)."""
    if torch is None:
        raise ImportError("PyTorch is required for get_device().")
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

def make_env(cfg: Config, seed_offset: int = 0, wrappers: Optional[List[Callable]] = None):
    """Create a seeded Gymnasium environment, applying optional wrappers in order."""
    if gym is None:
        raise ImportError("gymnasium is required for make_env(). pip install gymnasium")
    kwargs: Dict[str, Any] = {}
    if cfg.render_mode:
        kwargs["render_mode"] = cfg.render_mode
    if cfg.max_episode_steps:
        kwargs["max_episode_steps"] = cfg.max_episode_steps

    env = gym.make(cfg.env_id, **kwargs)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    for wrap in wrappers or []:
        env = wrap(env)

    seed = cfg.seed + seed_offset
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def space_dims(env) -> Tuple[Tuple[int, ...], int]:
    """Return (observation shape, number of discrete actions)."""
    obs_shape = env.observation_space.shape
    if not hasattr(env.action_space, "n"):
        raise ValueError("space_dims() assumes a discrete action space.")
    return obs_shape, int(env.action_space.n)


# --------------------------------------------------------------------------- #
# Experience replay
# --------------------------------------------------------------------------- #

class ReplayBuffer:
    """Fixed-size circular buffer of transitions stored as NumPy arrays."""

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], obs_dtype=np.float32):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done) -> None:
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def sample(self, batch_size: int, device=None) -> Dict[str, Any]:
        """Sample a batch. Returns torch tensors if a device is given, else NumPy arrays."""
        idx = np.random.randint(0, len(self), size=batch_size)
        batch = {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }
        if device is not None:
            if torch is None:
                raise ImportError("PyTorch is required to sample tensors.")
            batch = {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
        return batch


# --------------------------------------------------------------------------- #
# Schedules
# --------------------------------------------------------------------------- #

class LinearSchedule:
    """Linearly interpolate from start to end over `duration` steps, then hold."""

    def __init__(self, start: float, end: float, duration: int):
        self.start, self.end, self.duration = start, end, max(1, duration)

    def __call__(self, step: int) -> float:
        frac = min(step / self.duration, 1.0)
        return self.start + frac * (self.end - self.start)


def epsilon_greedy(q_values, epsilon: float, n_actions: int) -> int:
    """Pick a random action with probability epsilon, else argmax of q_values."""
    if random.random() < epsilon:
        return random.randrange(n_actions)
    if torch is not None and isinstance(q_values, torch.Tensor):
        return int(q_values.argmax().item())
    return int(np.argmax(q_values))


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

class Logger:
    """Console + CSV logger with a rolling window of recent episode returns."""

    def __init__(self, out_dir: os.PathLike, window: int = 100):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "metrics.csv"
        self._writer = None
        self._file = None
        self._fields: Optional[List[str]] = None
        self.returns: Deque[float] = deque(maxlen=window)
        self.lengths: Deque[int] = deque(maxlen=window)
        self.start_time = time.time()

    def record_episode(self, episode_return: float, episode_length: int) -> None:
        self.returns.append(float(episode_return))
        self.lengths.append(int(episode_length))

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.returns)) if self.returns else float("nan")

    def log(self, step: int, **metrics: float) -> None:
        row = {
            "step": step,
            "time_s": round(time.time() - self.start_time, 1),
            "mean_return": round(self.mean_return, 3),
            **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in metrics.items()},
        }
        if self._writer is None:
            self._fields = list(row.keys())
            self._file = open(self.csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fields, extrasaction="ignore")
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()
        print(" | ".join(f"{k}={v}" for k, v in row.items()))

    def close(self) -> None:
        if self._file:
            self._file.close()


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #

def save_checkpoint(path: os.PathLike, step: int, model, optimizer=None, **extra: Any) -> Path:
    """Save model/optimizer state plus any extra metadata."""
    if torch is None:
        raise ImportError("PyTorch is required for checkpoints.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"step": step, "model": model.state_dict(), **extra}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)
    return path


def load_checkpoint(path: os.PathLike, model, optimizer=None, device=None) -> Dict[str, Any]:
    """Load a checkpoint into model (and optimizer). Returns the full state dict."""
    if torch is None:
        raise ImportError("PyTorch is required for checkpoints.")
    state = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate(env, policy: Callable[[np.ndarray], int], episodes: int = 10,
             seed: Optional[int] = None) -> Dict[str, float]:
    """Run `policy` greedily for N episodes. Returns mean/std/min/max return and mean length."""
    returns, lengths = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=None if seed is None else seed + ep)
        done, total, steps = False, 0.0, 0
        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
            done = terminated or truncated
        returns.append(total)
        lengths.append(steps)
    return {
        "eval_mean": float(np.mean(returns)),
        "eval_std": float(np.std(returns)),
        "eval_min": float(np.min(returns)),
        "eval_max": float(np.max(returns)),
        "eval_len": float(np.mean(lengths)),
    }


__all__ = [
    "Config", "set_seed", "get_device", "make_env", "space_dims",
    "ReplayBuffer", "LinearSchedule", "epsilon_greedy",
    "Logger", "save_checkpoint", "load_checkpoint", "evaluate",
]