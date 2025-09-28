import numpy as np
import pandas as pd
from typing import Iterable
from config import EPS, VERY_NEGATIVE_SV, BIN2RANK


def nz(x):
    return np.minimum(1.0, np.maximum(0.0, x))


def relu_minus(x):
    return np.maximum(0.0, -x)


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def to_percent(x):
    return nz(np.asarray(x, dtype="float64") / 100.0)


def safe_nan(x):
    x = pd.to_numeric(x, errors="coerce")
    x[(x <= VERY_NEGATIVE_SV)] = np.nan
    return x


def map_bin_to_rank(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    return s.map(BIN2RANK).fillna(pd.to_numeric(s, errors="coerce")).astype(float)


def group_roll_median(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    return x.rolling(window=window, min_periods=min_periods).median()


def group_roll_mad(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    med = group_roll_median(x, window, min_periods)
    mad = (x - med).abs().rolling(window=window, min_periods=min_periods).median()
    return 1.4826 * mad


def robust_z(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    med = group_roll_median(x, window, min_periods)
    mad = group_roll_mad(x, window, min_periods)
    return (x - med) / (mad + EPS)


def mom(x: pd.Series) -> pd.Series:
    return x - x.shift(1)


def hhi(weights: Iterable) -> float:
    w = np.asarray(weights, dtype="float64")
    s = w / (w.sum() + EPS)
    return float(np.sum(s * s))


def as_month_sorted(df: pd.DataFrame, ym_col: str):
    df = df.copy()
    df[ym_col] = pd.to_datetime(df[ym_col].astype(str), format="%Y%m")
    return df.sort_values([ym_col])