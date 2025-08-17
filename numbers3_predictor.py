#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
numbers3_predictor_mod.py
-------------------------
User's existing `numbers3_predictor.py` への“非破壊”拡張版。
- 既存の関数名（preprocess_data / create_advanced_features など）を尊重しつつ、
  追加の高性能特徴量と Stacking 学習・時系列CVを提供
- 既存コードから import されても、単体スクリプトとしても動作

依存: numpy, pandas, scikit-learn, joblib
"""

from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from joblib import dump, load

RNG_SEED = 42
np.random.seed(RNG_SEED)


# =============
# I/O utilities
# =============

def _parse_numbers_cell(x: Any) -> List[int]:
    """頑健に [1,2,3] を返すパーサ"""
    if isinstance(x, list):
        return [int(v) for v in x][:3]
    if isinstance(x, (tuple, set)):
        return [int(v) for v in list(x)][:3]
    if isinstance(x, str):
        s = x
        for ch in "[](){}":
            s = s.replace(ch, " ")
        s = s.replace(",", " ").replace("　", " ")
        toks = [t for t in s.split() if t.isdigit()]
        out = [int(t) for t in toks]
        return out[:3]
    if np.isscalar(x):
        return [int(x)]
    return []


def load_history(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 列名ゆらぎ対応
    date_col = None
    for c in ["抽せん日", "date", "Date", "日付"]:
        if c in df.columns:
            date_col = c
            break
    nums_col = None
    for c in ["本数字", "numbers", "当選数字", "winning_numbers"]:
        if c in df.columns:
            nums_col = c
            break
    if date_col is None or nums_col is None:
        raise ValueError("CSVには '抽せん日' と '本数字'（または同義列）が必要です。")

    df = df[[date_col, nums_col]].copy()
    df.rename(columns={date_col: "抽せん日", nums_col: "本数字"}, inplace=True)
    df["抽せん日"] = pd.to_datetime(df["抽せん日"])
    df.sort_values("抽せん日", inplace=True)
    df["本数字"] = df["本数字"].map(_parse_numbers_cell)
    df = df[df["本数字"].map(len) == 3].reset_index(drop=True)
    if df.empty:
        raise ValueError("有効な3桁データがありません。CSVを確認してください。")
    return df


# ========================
# Advanced Feature Set v2
# ========================

@dataclass
class WindowConfig:
    window: int = 60           # 直近の統計
    prev_window: int = 60      # トレンド比較用
    pair_window: int = 120     # ペア・トリプル統計窓
    gap_cap: int = 120         # 欠番間隔の最大値クリップ


def _counts_lastN(history: List[List[int]], N: int) -> np.ndarray:
    counts = np.zeros(10, dtype=float)
    for draw in history[-N:]:
        for d in draw:
            if 0 <= d <= 9:
                counts[d] += 1.0
    return counts


def _gaps(history: List[List[int]], N: int, cap: int) -> np.ndarray:
    gaps = np.full(10, cap, dtype=float)
    for idx_back, draw in enumerate(reversed(history[-N:]), start=1):
        for d in set(draw):
            gaps[d] = min(gaps[d], idx_back - 1)
    return gaps


def _co_occurrence_pairs(history: List[List[int]], N: int) -> np.ndarray:
    """0-9 のペア(10x10上三角)の共起回数（対称行列として返す）"""
    mat = np.zeros((10, 10), dtype=float)
    for draw in history[-N:]:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = draw[i], draw[j]
                if 0 <= a <= 9 and 0 <= b <= 9 and a != b:
                    mat[a, b] += 1.0
                    mat[b, a] += 1.0
    return mat


def _co_occurrence_triples(history: List[List[int]], N: int) -> np.ndarray:
    """3つ同時出現のカウント（ダイアゴナルは常に0） -> 各数字の「共起度合い」へ要約"""
    # 各数字が「他と一緒に」出やすい傾向を 10次元に要約（単純にトリプル回数を数字ごと合算）
    tri = np.zeros(10, dtype=float)
    for draw in history[-N:]:
        s = list(set(draw))
        if len(s) == 3:
            for d in s:
                tri[d] += 1.0
    return tri


def _entropy_from_counts(counts: np.ndarray) -> float:
    p = counts / max(counts.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0


def _trend(curr_counts: np.ndarray, prev_counts: np.ndarray) -> float:
    curr = curr_counts / max(curr_counts.sum(), 1.0)
    prev = prev_counts / max(prev_counts.sum(), 1.0)
    return float(np.abs(curr - prev).sum())


def build_dataset_v2(df: pd.DataFrame, cfg: WindowConfig):
    """
    t 時点の特徴量は「t 以前の履歴のみ」を使用（リーク防止）
    特徴:
      - freq(10), inv_gap(10), pair_strength(10), triple_strength(10),
        entropy(1), trend(1), periodic_sin(1), periodic_cos(1) = 44 次元
    ラベル: t の当選数字の multi-hot(10)
    """
    W = max(cfg.window, cfg.prev_window, cfg.pair_window)
    X_list, Y_list, TS = [], [], []

    seq = df["本数字"].tolist()
    tss = df["抽せん日"].tolist()

    for t in range(W, len(df)):
        hist = seq[:t]

        counts = _counts_lastN(hist, cfg.window)
        prev_counts = _counts_lastN(hist[:-cfg.window] if len(hist) > cfg.window else hist, cfg.prev_window)
        gaps = _gaps(hist, cfg.window, cfg.gap_cap)
        pairs = _co_occurrence_pairs(hist, cfg.pair_window).sum(axis=1)  # 各数字の共起強度へ要約
        triples = _co_occurrence_triples(hist, cfg.pair_window)

        freq = counts / max(counts.sum(), 1.0)
        inv_gap = 1.0 / (1.0 + gaps)
        pair_strength = pairs / max(pairs.sum(), 1.0)
        triple_strength = triples / max(triples.sum(), 1.0)
        entropy = _entropy_from_counts(counts)
        tr = _trend(counts, prev_counts)
        sin_t = math.sin(2 * math.pi * (t % 7) / 7.0)
        cos_t = math.cos(2 * math.pi * (t % 7) / 7.0)

        feats = np.concatenate([
            freq, inv_gap, pair_strength, triple_strength, [entropy, tr, sin_t, cos_t]
        ]).astype(np.float32)
        X_list.append(feats)

        y = np.zeros(10, dtype=int)
        for d in seq[t]:
            if 0 <= d <= 9:
                y[d] = 1
        Y_list.append(y)
        TS.append(tss[t])

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    return X, Y, TS


# ================================
# Model: Multi-label Stacking v2
# ================================

def build_stacking_model(random_state: int = RNG_SEED) -> MultiOutputClassifier:
    base = [
        ("rf", RandomForestClassifier(
            n_estimators=500, random_state=random_state, n_jobs=-1
        )),
        ("et", ExtraTreesClassifier(
            n_estimators=700, random_state=random_state, n_jobs=-1
        )),
        ("gb", GradientBoostingClassifier(
            learning_rate=0.05, n_estimators=400, max_depth=3, random_state=random_state
        )),
        ("lr", LogisticRegression(C=1.2, max_iter=3000, solver="lbfgs"))
    ]
    final_est = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")
    stack = StackingClassifier(
        estimators=base,
        final_estimator=final_est,
        stack_method="predict_proba",
        passthrough=True,
        n_jobs=-1
    )
    clf = MultiOutputClassifier(stack, n_jobs=-1)
    return clf


@dataclass
class Artifacts:
    scaler: MinMaxScaler
    model: MultiOutputClassifier
    feature_dim: int


def time_series_cv(X: np.ndarray, Y: np.ndarray, splits: int = 5) -> Dict[str, float]:
    tscv = TimeSeriesSplit(n_splits=splits)
    f1s, ps, rs = [], [], []
    for tr, va in tscv.split(X):
        Xtr, Xva = X[tr], X[va]
        Ytr, Yva = Y[tr], Y[va]

        sc = MinMaxScaler()
        Xtr = sc.fit_transform(Xtr)
        Xva = sc.transform(Xva)

        mdl = build_stacking_model()
        mdl.fit(Xtr, Ytr)

        proba = np.stack([p[:, 1] for p in mdl.predict_proba(Xva)], axis=1)
        top3 = np.argsort(-proba, axis=1)[:, :3]

        Yhat = np.zeros_like(Yva)
        for i, idxs in enumerate(top3):
            Yhat[i, idxs] = 1

        f1s.append(f1_score(Yva, Yhat, average="macro", zero_division=0))
        ps.append(precision_score(Yva, Yhat, average="macro", zero_division=0))
        rs.append(recall_score(Yva, Yhat, average="macro", zero_division=0))
    return {"f1_macro": float(np.mean(f1s)), "precision_macro": float(np.mean(ps)), "recall_macro": float(np.mean(rs))}


def train_all(X: np.ndarray, Y: np.ndarray) -> Artifacts:
    sc = MinMaxScaler()
    Xs = sc.fit_transform(X)
    mdl = build_stacking_model()
    mdl.fit(Xs, Y)
    return Artifacts(scaler=sc, model=mdl, feature_dim=X.shape[1])


# =====================
# Inference / Candidates
# =====================

def rank_digits(proba_vec: np.ndarray) -> List[int]:
    return list(np.argsort(-proba_vec))


def score_combinations(proba_vec: np.ndarray, k: int = 3, top_m: int = 20, mode: str = "sum"):
    combos = list(combinations(range(10), k))
    scored = []
    for c in combos:
        ps = [proba_vec[d] for d in c]
        val = float(np.prod(ps)) if mode == "product" else float(np.sum(ps))
        scored.append((list(c), val))
    scored.sort(key=lambda z: z[1], reverse=True)
    return scored[:top_m]


def predict_next(art: Artifacts, X_last: np.ndarray, k: int = 3, top_m: int = 20) -> Dict[str, Any]:
    Xs = art.scaler.transform(X_last.reshape(1, -1))
    proba_list = art.model.predict_proba(Xs)
    proba = np.stack([p[:, 1] for p in proba_list], axis=1).ravel()

    return {
        "digit_probabilities": {i: float(proba[i]) for i in range(10)},
        "top_k_digits": rank_digits(proba)[:k],
        "top_combinations": [{"combo": c, "score": s} for c, s in score_combinations(proba, k=k, top_m=top_m)]
    }


# ========
#   CLI
# ========

def main():
    ap = argparse.ArgumentParser(description="numbers3 predictor - mod version (stacking & advanced features)")
    ap.add_argument("--csv", type=str, required=True, help="履歴CSVのパス（列: 抽せん日, 本数字）")
    ap.add_argument("--window", type=int, default=60, help="直近の統計窓")
    ap.add_argument("--prev-window", type=int, default=60, help="トレンド比較用窓")
    ap.add_argument("--pair-window", type=int, default=120, help="ペア/トリプル共起の窓")
    ap.add_argument("--gap-cap", type=int, default=120, help="欠番間隔の上限クリップ")
    ap.add_argument("--cv-splits", type=int, default=5, help="TimeSeriesSplit 分割数")
    ap.add_argument("--k", type=int, default=3, choices=[3, 4], help="候補に使う桁数")
    ap.add_argument("--top-m", type=int, default=20, help="上位組合せの出力数")
    ap.add_argument("--model-out", type=str, default="numbers3_mod_artifacts.joblib", help="学習済みの保存先")
    ap.add_argument("--pred-out", type=str, default="numbers3_mod_candidates.json", help="予測出力(JSON)")
    args = ap.parse_args()

    # 1) データ読込 & 特徴量
    df = load_history(args.csv)
    cfg = WindowConfig(window=args.window, prev_window=args.prev_window, pair_window=args.pair_window, gap_cap=args.gap_cap)
    X, Y, TS = build_dataset_v2(df, cfg)
    print(f"[INFO] Dataset: X={X.shape} Y={Y.shape} samples={len(TS)} (from {TS[0].date()} to {TS[-1].date()})")

    # 2) 時系列CV
    scores = time_series_cv(X, Y, splits=args.cv_splits)
    print("[CV] macro-F1={f1_macro:.4f}  Precision={precision_macro:.4f}  Recall={recall_macro:.4f}".format(**scores))

    # 3) 全データで学習
    art = train_all(X, Y)

    # 保存
    if args.model_out:
        dump(art, args.model_out)
        print(f"[SAVE] Artifacts -> {args.model_out}")

    # 4) 直近特徴量で次回予測
    result = predict_next(art, X[-1], k=args.k, top_m=args.top_m)
    if args.pred_out:
        with open(args.pred_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] Candidates -> {args.pred_out}")

    # 表示
    print("\n=== Next draw (mod) ===")
    print("Digit Probabilities:")
    for d, p in result["digit_probabilities"].items():
        print(f"  {d}: {p:.4f}")
    print(f"Top-{args.k} digits:", result["top_k_digits"])
    print("Top combinations:")
    for row in result["top_combinations"]:
        print(f"  {row['combo']}  score={row['score']:.4f}")


if __name__ == "__main__":
    main()
