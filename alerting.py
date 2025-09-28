import numpy as np
import pandas as pd
from config import THRESHOLDS
from utils import nz


def rolling_mean(x: pd.Series, k: int) -> pd.Series:
    return x.rolling(window=k, min_periods=1).mean()


def assign_alert(prob: pd.Series) -> pd.Series:
    t_y, t_o, t_r, delta = THRESHOLDS["yellow"], THRESHOLDS["orange"], THRESHOLDS["red"], THRESHOLDS["delta"]
    k = THRESHOLDS["persistence_k"]

    pbar = rolling_mean(prob, k)
    lbl = np.where((prob >= t_r) & (pbar >= t_r - delta), "RED",
                   np.where(prob >= t_o, "ORANGE",
                            np.where(prob >= t_y, "YELLOW", "GREEN")))
    return pd.Series(lbl, index=prob.index)