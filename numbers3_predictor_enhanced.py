# numbers3_predictor_enhanced.py
# -*- coding: utf-8 -*-
"""
Enhanced Numbers3 predictor:
- Unified-reward PPO environment (accuracy/diversity/cycle/profit)
- Stronger feature engineering for time-series
- TimeSeriesSplit evaluation pipeline
- Missing utilities implemented (cycle score, prize classification, etc.)
- Consistent TOP_K handling and safer defaults
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Optional heavy deps (only needed for RL). Wrapped to avoid import errors when not installed.
try:
    import gym
    from gym import spaces
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover
    gym = None
    spaces = None
    PPO = None

SEED = 42
TOP_K = 3  # Numbers3: choose 3 digits per draw

def set_global_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)

set_global_seed()


# -------------------------
# Data utilities & features
# -------------------------
def _ensure_list_of_ints(x) -> List[int]:
    if isinstance(x, list):
        return [int(v) for v in x]
    if isinstance(x, str):
        cleaned = x.strip("[]").replace(",", " ").replace("'", "").replace('"', "")
        return [int(n) for n in cleaned.split() if n.isdigit()]
    return [int(x)] if pd.notna(x) else []


def create_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple numeric statistics derived from '本数字' (list of 3 digits) and date '抽せん日'."""
    data = df.copy()
    data['本数字'] = data['本数字'].apply(_ensure_list_of_ints)
    data['抽せん日'] = pd.to_datetime(data['抽せん日'])
    data = data[data['本数字'].apply(len) == 3].sort_values('抽せん日').reset_index(drop=True)

    nums_array = np.vstack(data['本数字'].values)  # shape (n, 3)
    data['数字合計'] = nums_array.sum(axis=1)
    data['数字平均'] = nums_array.mean(axis=1)
    data['最大'] = nums_array.max(axis=1)
    data['最小'] = nums_array.min(axis=1)
    data['標準偏差'] = np.std(nums_array, axis=1)
    return data


def _rolling_counts_of_digits(series_of_lists: Iterable[List[int]], window: int) -> pd.DataFrame:
    """Compute rolling counts (0..9) over the last `window` draws."""
    counts = np.zeros((len(series_of_lists), 10), dtype=float)
    hist = Counter()
    deque_buffer: List[List[int]] = []
    for i, lst in enumerate(series_of_lists):
        deque_buffer.append(lst)
        for d in lst:
            if 0 <= d <= 9:
                hist[d] += 1
        # drop old
        if len(deque_buffer) > window:
            for old_d in deque_buffer.pop(0):
                if 0 <= old_d <= 9:
                    hist[old_d] -= 1
        # snapshot
        for d in range(10):
            counts[i, d] = hist[d]
    cols = [f'roll{window}_digit{d}' for d in range(10)]
    return pd.DataFrame(counts, columns=cols)


def _gaps_since_last_seen(series_of_lists: Iterable[List[int]]) -> pd.DataFrame:
    """For each digit 0..9, distance (in draws) since last occurrence (∞ -> large number)."""
    gaps = np.zeros((len(series_of_lists), 10), dtype=float)
    last_pos = {d: -1 for d in range(10)}
    for i, lst in enumerate(series_of_lists):
        for d in range(10):
            gaps[i, d] = (i - last_pos[d]) if last_pos[d] >= 0 else 100.0
        for d in lst:
            if 0 <= d <= 9:
                last_pos[d] = i
    cols = [f'gap_digit{d}' for d in range(10)]
    return pd.DataFrame(gaps, columns=cols)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data['idx'] = np.arange(len(data))
    data['dow'] = data['抽せん日'].dt.dayofweek
    data['month'] = data['抽せん日'].dt.month
    # rolling digit counts
    roll10 = _rolling_counts_of_digits(data['本数字'], window=10)
    roll30 = _rolling_counts_of_digits(data['本数字'], window=30)
    gaps = _gaps_since_last_seen(data['本数字'])
    data = pd.concat([data, roll10, roll30, gaps], axis=1)
    return data


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, pd.DataFrame]:
    """Return X (scaled), Y (multi-label 10-dim), scaler, and enriched DataFrame."""
    base = create_base_features(df)
    feats = add_time_features(base)

    # Binary labels (presence of digit) – duplicates collapse into 1
    y = np.zeros((len(feats), 10), dtype=int)
    for i, lst in enumerate(feats['本数字']):
        for d in lst:
            if 0 <= d <= 9:
                y[i, d] = 1

    num_cols = feats.select_dtypes(include=[np.number]).columns
    X = feats[num_cols].values
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)
    return Xs, y, scaler, feats


# -------------------------
# Evaluation
# -------------------------
def time_series_cv_f1(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> float:
    """
    Train one-vs-rest LogisticRegression for each digit with TimeSeriesSplit.
    Returns micro-averaged F1 across splits.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    f1s = []
    for train_idx, test_idx in tscv.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        preds = np.zeros_like(yte)
        for d in range(10):
            clf = LogisticRegression(max_iter=200, solver='saga', penalty='l2', n_jobs=None)
            clf.fit(Xtr, ytr[:, d])
            p = clf.predict(Xte)
            preds[:, d] = p
        f1s.append(f1_score(yte.ravel(), preds.ravel(), average='micro'))
    return float(np.mean(f1s))


# -------------------------
# Lottery utilities
# -------------------------
def classify_numbers3_prize(selected: List[int], winning: List[int]) -> str:
    """Rudimentary Numbers3 prize classifier. (Domain assumptions)"""
    if len(selected) != 3 or len(winning) != 3:
        return "はずれ"
    if selected == winning:
        return "ストレート"
    if Counter(selected) == Counter(winning):
        return "ボックス"
    if selected[1:] == winning[1:]:  # last 2 digits exact match
        return "ミニ"
    return "はずれ"


def calculate_number_cycle_score(historical_numbers: List[List[int]]) -> Dict[int, float]:
    """
    Compute a 'cycle score' per digit based on average gap since last seen.
    Lower is 'hotter'. If never seen, fallback to large value.
    """
    last_seen = {d: None for d in range(10)}
    gaps_per_digit = defaultdict(list)
    for i, lst in enumerate(historical_numbers):
        for d in range(10):
            if last_seen[d] is not None:
                gaps_per_digit[d].append(i - last_seen[d])
        for d in lst:
            if 0 <= d <= 9:
                last_seen[d] = i
    scores = {}
    for d in range(10):
        if gaps_per_digit[d]:
            scores[d] = float(np.mean(gaps_per_digit[d]))
        else:
            scores[d] = 999.0
    return scores


# -------------------------
# Unified-reward PPO Env
# -------------------------
@dataclass
class RewardWeights:
    accuracy: float = 0.6
    diversity: float = 0.1
    cycle: float = 0.15
    profit: float = 0.15


class UnifiedNumbers3Env:
    """
    A single environment that optimizes a unified reward:
    - accuracy: overlap with recent real draws (proxy)
    - diversity: penalize repeating the same combo
    - cycle: prefer lower cycle-score digits (recent/hot)
    - profit: reward mapping based on prize type
    """

    def __init__(self, historical_numbers: List[List[int]], weights: RewardWeights | None = None, top_k: int = TOP_K):
        if gym is None or spaces is None:
            raise RuntimeError("Gym / spaces not available. Install gymnasium/gym and stable-baselines3 to use RL.")
        self.historical_numbers = historical_numbers
        self.top_k = top_k
        self.weights = weights or RewardWeights()
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        self.previous_outputs = set()
        self.cycle_scores = calculate_number_cycle_score(historical_numbers)

    def reset(self):
        return np.zeros(10, dtype=np.float32)

    def _select_topk(self, action: np.ndarray) -> List[int]:
        idxs = np.argsort(action)[-self.top_k:]
        return sorted(idxs.tolist())

    def _accuracy_reward(self, selected: List[int]) -> float:
        # proxy: overlap with most recent historical draw
        target = set(self.historical_numbers[-1])
        return len(set(selected) & target) / self.top_k

    def _diversity_reward(self, selected: Tuple[int, ...]) -> float:
        if selected in self.previous_outputs:
            return -1.0
        return 1.0

    def _cycle_reward(self, selected: List[int]) -> float:
        avg_cycle = np.mean([self.cycle_scores.get(d, 999.0) for d in selected])
        return max(0.0, 1.0 - (avg_cycle / 50.0))

    def _profit_reward(self, selected: List[int]) -> float:
        # best payoff across history
        table = {"ストレート": 1.0, "ボックス": 0.25, "ミニ": 0.1, "はずれ": 0.0}
        best = 0.0
        for w in self.historical_numbers[-100:]:  # limit for speed
            best = max(best, table[classify_numbers3_prize(selected, w)])
        return best

    def step(self, action):
        action = np.asarray(action).reshape(-1)
        if action.size != 10:
            return np.zeros(10, dtype=np.float32), -1.0, True, {}

        selected = tuple(self._select_topk(action))
        acc = self._accuracy_reward(list(selected))
        div = self._diversity_reward(selected)
        cyc = self._cycle_reward(list(selected))
        prof = self._profit_reward(list(selected))

        reward = (
            self.weights.accuracy * acc
            + self.weights.diversity * div
            + self.weights.cycle * cyc
            + self.weights.profit * prof
        )
        self.previous_outputs.add(selected)
        obs = np.zeros(10, dtype=np.float32)
        done = True
        info = {"acc": acc, "div": div, "cycle": cyc, "profit": prof}
        return obs, float(reward), done, info


# -------------------------
# Candidate generation
# -------------------------
def estimate_digit_probabilities(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit simple one-vs-rest logistic models on the *entire* data for producing
    per-digit probabilities on the last row (most recent state). For production,
    you should fit on train and predict on test; this is a lightweight utility
    to create probabilities for candidate generation.
    """
    probs = np.zeros(10, dtype=float)
    if len(X) == 0:
        return probs
    last = X[-1:]
    for d in range(10):
        clf = LogisticRegression(max_iter=200, solver='saga', penalty='l2')
        clf.fit(X, y[:, d])
        p = float(clf.predict_proba(last)[0, 1])
        probs[d] = p
    # normalize
    s = probs.sum()
    if s > 0:
        probs = probs / s
    return probs


def generate_top_combinations(prob: np.ndarray, top_k: int = TOP_K, allow_duplicates: bool = True, top_n: int = 50) -> List[Tuple[List[int], float]]:
    """
    Make candidate digit combinations by greedy beam search on independent per-digit probabilities.
    Score = product of probs (or adjusted for duplicates if not allowed).
    """
    digits = list(range(10))
    if allow_duplicates:
        # simple sampling with replacement, sorted for stability
        # generate a broad set and keep top_n unique by score
        candidates = {}
        for a in digits:
            for b in digits:
                for c in digits:
                    combo = [a, b, c]
                    score = prob[a] * prob[b] * prob[c]
                    key = tuple(combo)
                    if key not in candidates or candidates[key] < score:
                        candidates[key] = score
        ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return [([a, b, c], float(s)) for (a, b, c), s in ranked]
    else:
        # combinations without replacement
        candidates = []
        for a in range(10):
            for b in range(a + 1, 10):
                for c in range(b + 1, 10):
                    score = prob[a] * prob[b] * prob[c]
                    candidates.append(([a, b, c], score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [(c, float(s)) for c, s in candidates[:top_n]]


# -------------------------
# Orchestration helpers
# -------------------------
def train_unified_ppo(historical_numbers: List[List[int]], total_timesteps: int = 5000, weights: RewardWeights | None = None):
    """Train a PPO agent on the unified environment and return it (requires stable-baselines3)."""
    if PPO is None or gym is None:
        raise RuntimeError("stable-baselines3 / gym not available.")
    env = UnifiedNumbers3Env(historical_numbers, weights=weights)
    model = PPO("MlpPolicy", env, verbose=0, seed=SEED)
    model.learn(total_timesteps=total_timesteps)
    return model


def pipeline_from_dataframe(df: pd.DataFrame, allow_duplicates: bool = True, top_n: int = 50) -> Dict[str, object]:
    """
    End-to-end lightweight pipeline:
    - feature build
    - time-series CV scoring (micro-F1)
    - digit probability estimation
    - candidate generation
    """
    X, y, scaler, enriched = build_feature_matrix(df)
    cv_f1 = time_series_cv_f1(X, y, n_splits=5) if len(X) >= 50 else np.nan
    prob = estimate_digit_probabilities(X, y)
    candidates = generate_top_combinations(prob, top_k=TOP_K, allow_duplicates=allow_duplicates, top_n=top_n)
    return {
        "cv_f1_micro": cv_f1,
        "digit_probabilities": prob,
        "candidates": candidates,
        "scaler": scaler,
        "enriched_df": enriched,
    }


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Example stub: expects a CSV with columns ['抽せん日','本数字'] where 本数字 is like "[1, 2, 3]"
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to historical CSV with 抽せん日, 本数字")
    parser.add_argument("--allow-duplicates", action="store_true", help="Allow repeated digits in candidates")
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()

    if args.csv is None:
        print("No CSV provided. Exiting.")
        exit(0)

    hist = pd.read_csv(args.csv)
    out = pipeline_from_dataframe(hist, allow_duplicates=args.allow_duplicates, top_n=args.top_n)
    print(json.dumps({
        "cv_f1_micro": None if np.isnan(out["cv_f1_micro"]) else out["cv_f1_micro"],
        "top5_candidates": out["candidates"][:5],
    }, ensure_ascii=False, indent=2))
