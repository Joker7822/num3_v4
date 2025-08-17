# numbers3_predictor.py (compat shim)
# -*- coding: utf-8 -*-
"""
Compatibility shim for legacy scripts (e.g., rerun_or_continue_numbers3.py)
- Uses Gymnasium if available; falls back to Gym.
- Provides bulk_predict_all_past_draws expected by legacy runner.
- Proxies to enhanced pipeline for feature building and candidate generation.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

# Prefer Gymnasium (Gym is unmaintained for NumPy 2.x)
try:
    import gymnasium as gym  # noqa: F401
    from gymnasium import spaces  # noqa: F401
except Exception:
    import gym  # type: ignore  # noqa: F401
    from gym import spaces  # type: ignore  # noqa: F401
    warnings.warn(
        "Gym is unmaintained; please switch to Gymnasium. "
        "This shim will still run if Gym is present, but compatibility is not guaranteed.",
        RuntimeWarning
    )

# Reuse enhanced pipeline
try:
    from numbers3_predictor_enhanced import (
        pipeline_from_dataframe,
        build_feature_matrix,
        time_series_cv_f1,
        estimate_digit_probabilities,
        generate_top_combinations,
        classify_numbers3_prize,
        calculate_number_cycle_score,
    )
except Exception as e:
    raise ImportError(
        "numbers3_predictor_enhanced.py is required by this shim. "
        "Please place it next to this file."
    ) from e


def bulk_predict_all_past_draws(df: pd.DataFrame,
                                allow_duplicates: bool = True,
                                top_n: int = 20,
                                min_history: int = 30):
    """
    Rolling backtest-style prediction for each draw i using data up to i-1.
    Returns a DataFrame with columns:
      抽せん日, 本数字, candidates(list of (combo,score)), digit_probabilities(np.array), cv_f1_micro
    """
    df = df.copy()
    if '抽せん日' not in df.columns or '本数字' not in df.columns:
        raise ValueError("Input DataFrame must have columns ['抽せん日','本数字']")
    df['抽せん日'] = pd.to_datetime(df['抽せん日'])
    df = df.sort_values('抽せん日').reset_index(drop=True)

    rows = []
    for i in range(len(df)):
        hist = df.iloc[:i]
        current = df.iloc[i]
        if len(hist) < min_history:
            rows.append({
                '抽せん日': current['抽せん日'],
                '本数字': current['本数字'],
                'candidates': [],
                'digit_probabilities': np.full(10, np.nan),
                'cv_f1_micro': np.nan,
            })
            continue
        out = pipeline_from_dataframe(hist, allow_duplicates=allow_duplicates, top_n=top_n)
        rows.append({
            '抽せん日': current['抽せん日'],
            '本数字': current['本数字'],
            'candidates': out['candidates'],
            'digit_probabilities': out['digit_probabilities'],
            'cv_f1_micro': out['cv_f1_micro'],
        })
    return pd.DataFrame(rows)


# Useful re-exports for legacy scripts
__all__ = [
    'bulk_predict_all_past_draws',
    'pipeline_from_dataframe',
    'build_feature_matrix',
    'time_series_cv_f1',
    'estimate_digit_probabilities',
    'generate_top_combinations',
    'classify_numbers3_prize',
    'calculate_number_cycle_score',
]
