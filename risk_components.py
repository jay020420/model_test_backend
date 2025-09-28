import numpy as np
import pandas as pd
from typing import Tuple
from config import ROLL_WINDOW, MIN_PERIODS, EPS
from utils import (
    nz, relu_minus, logistic, to_percent, robust_z, mom
)


def compute_sales_risk(df: pd.DataFrame, key=["ENCODED_MCT", "TA_YM"]) -> pd.DataFrame:
    d = df.sort_values(key).copy()

    rS = d["RC_M1_SAA_RANK"]
    rC = d["RC_M1_TO_UE_CT_RANK"]
    rA = d["RC_M1_AV_NP_AT_RANK"]
    rX = d["APV_CE_RAT_RANK"]

    s_drop = nz((relu_minus(mom(rS)) + relu_minus(mom(rC))) / 2.0)
    s_aov = nz(relu_minus(robust_z(rA, ROLL_WINDOW, MIN_PERIODS)))
    s_cxl = nz(np.maximum(rX, logistic(robust_z(rX, ROLL_WINDOW, MIN_PERIODS))))

    ind_rank = nz(1.0 - to_percent(d.get("M12_SME_RY_SAA_PCE_RT", 0)))
    bzn_rank = nz(1.0 - to_percent(d.get("M12_SME_BZN_SAA_PCE_RT", 0)))
    s_peer = nz((ind_rank + bzn_rank) / 2.0)

    dlv = nz(to_percent(d.get("DLV_SAA_RAT", 0)))
    s_dlv_jump = nz(relu_minus(-mom(dlv)) + relu_minus(-robust_z(dlv, ROLL_WINDOW, MIN_PERIODS)))

    sales_risk = (
            0.35 * s_drop +
            0.15 * s_aov +
            0.20 * s_cxl +
            0.20 * s_peer +
            0.10 * (dlv * s_dlv_jump)
    )

    out = d[["ENCODED_MCT", "TA_YM"]].copy()
    out["Sales_Risk"] = nz(sales_risk)
    return out


def compute_customer_risk(df: pd.DataFrame, key=["ENCODED_MCT", "TA_YM"]) -> pd.DataFrame:
    d = df.sort_values(key).copy()

    rU = d["RC_M1_UE_CUS_CN_RANK"]
    s_cus_drop = nz(relu_minus(mom(rU)) + relu_minus(robust_z(rU, ROLL_WINDOW, MIN_PERIODS)))

    q_reu = nz(to_percent(d.get("MCT_UE_CLN_REU_RAT", 0)))
    q_new = nz(to_percent(d.get("MCT_UE_CLN_NEW_RAT", 0)))
    s_loyal = nz(relu_minus(mom(q_reu)))
    s_acq = nz(relu_minus(mom(q_new)))

    cols_age_f = [
        "M12_MAL_1020_RAT", "M12_MAL_30_RAT", "M12_MAL_40_RAT", "M12_MAL_50_RAT", "M12_MAL_60_RAT",
        "M12_FME_1020_RAT", "M12_FME_30_RAT", "M12_FME_40_RAT", "M12_FME_50_RAT", "M12_FME_60_RAT"
    ]
    w = nz(np.nan_to_num(d[cols_age_f].to_numpy(dtype="float64") / 100.0))
    hhi_age = (w * w).sum(axis=1)
    H_age = pd.Series(hhi_age, index=d.index)

    v = nz(np.nan_to_num(
        d[["RC_M1_SHC_RSD_UE_CLN_RAT", "RC_M1_SHC_WP_UE_CLN_RAT", "RC_M1_SHC_FLP_UE_CLN_RAT"]].to_numpy(
            dtype="float64") / 100.0
    ))
    hhi_type = (v * v).sum(axis=1)
    H_type = pd.Series(hhi_type, index=d.index)

    def pos_norm(x: pd.Series) -> pd.Series:
        z = (x - x.rolling(ROLL_WINDOW, MIN_PERIODS).median()) / (
                (x - x.rolling(ROLL_WINDOW, MIN_PERIODS).median()).abs().rolling(ROLL_WINDOW,
                                                                                 MIN_PERIODS).median() + EPS
        )
        return nz(np.maximum(0.0, z))

    s_mix = pos_norm(H_age)
    s_type = pos_norm(H_type)

    customer_risk = (
            0.40 * s_cus_drop +
            0.25 * s_loyal +
            0.20 * s_acq +
            0.10 * s_mix +
            0.05 * s_type
    )

    out = d[["ENCODED_MCT", "TA_YM"]].copy()
    out["Customer_Risk"] = nz(customer_risk)
    return out


def compute_market_risk(df: pd.DataFrame, key=["ENCODED_MCT", "TA_YM"]) -> pd.DataFrame:
    d = df.sort_values(key).copy()

    h_ind = nz(to_percent(d.get("M12_SME_RY_ME_MCT_RAT", 0)))
    h_bzn = nz(to_percent(d.get("M12_SME_BZN_ME_MCT_RAT", 0)))
    s_closure_env = (h_ind + h_bzn) / 2.0

    u_rev = nz(1.0 - to_percent(d.get("M1_SME_RY_SAA_RAT", 0)))
    u_cnt = nz(1.0 - to_percent(d.get("M1_SME_RY_CNT_RAT", 0)))
    s_underperf = (u_rev + u_cnt) / 2.0

    a = d["MCT_OPE_MS_CN_RANK"]
    s_age = nz(4.0 * np.minimum(a, 1.0 - a))

    market_risk = 0.50 * s_closure_env + 0.35 * s_underperf + 0.15 * s_age

    out = d[["ENCODED_MCT", "TA_YM"]].copy()
    out["Market_Risk"] = nz(market_risk)
    return out