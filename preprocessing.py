import pandas as pd
from config import MIN_PERIODS, ROLL_WINDOW
from utils import safe_nan, map_bin_to_rank, as_month_sorted

KEY_MCT = "ENCODED_MCT"
KEY_YM = "TA_YM"


def load_and_join(ds1: pd.DataFrame, ds2: pd.DataFrame, ds3: pd.DataFrame) -> pd.DataFrame:
    a = ds1.copy()
    b = ds2.copy()
    c = ds3.copy()

    for col in b.columns:
        if col not in (KEY_MCT, KEY_YM):
            b[col] = safe_nan(b[col])
    for col in c.columns:
        if col not in (KEY_MCT, KEY_YM):
            c[col] = safe_nan(c[col])

    b = as_month_sorted(b, KEY_YM)
    c = as_month_sorted(c, KEY_YM)

    df = b.merge(c, on=[KEY_MCT, KEY_YM], how="left", suffixes=("", "_C"))
    df = df.merge(a, on=[KEY_MCT], how="left")

    return df


def normalize_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bin_cols = [
        "RC_M1_SAA", "RC_M1_TO_UE_CT", "RC_M1_UE_CUS_CN",
        "RC_M1_AV_NP_AT", "APV_CE_RAT", "MCT_OPE_MS_CN"
    ]
    for col in bin_cols:
        if col in df.columns:
            df[col + "_RANK"] = map_bin_to_rank(df[col])
    return df