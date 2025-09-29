import pandas as pd
from typing import Optional
from preprocessing import load_and_join, normalize_bins
from risk_aggregate import compute_all_risks
from ensemble import weighted_ensemble, Calibrator
from alerting import assign_alert
from config import LAMBDA_BLEND


def run_pipeline(ds1: pd.DataFrame, ds2: pd.DataFrame, ds3: pd.DataFrame,
                 preds: Optional[pd.DataFrame] = None,
                 calib_fit_y: Optional[pd.Series] = None) -> pd.DataFrame:
    df = load_and_join(ds1, ds2, ds3)
    df = normalize_bins(df)

    risks = compute_all_risks(df)

    if preds is None:
        risks["p_model"] = 0.0
        p_cal = risks["p_model"]
    else:
        preds_idx = preds[["ENCODED_MCT", "TA_YM"]]
        risks = risks.merge(preds, on=["ENCODED_MCT", "TA_YM"], how="left")
        risks["p_model"] = weighted_ensemble(risks)
        cal = Calibrator()
        if calib_fit_y is not None:
            cal.fit(risks["p_model"].values, calib_fit_y.values)
        p_cal = cal.transform(risks["p_model"].values)
        risks["p_model_cal"] = p_cal

    risks["p_model"] = risks.get("p_model", 0).fillna(0)
    p_cal = risks.get("p_model_cal", risks["p_model"]).fillna(0)
    risks["p_final"] = (LAMBDA_BLEND * p_cal) + ((1 - LAMBDA_BLEND) * risks["RiskScore"].fillna(0))
    risks["p_final"] = risks["p_final"].fillna(0)
    risks["Alert"] = assign_alert(risks["p_final"].fillna(0))

    cols = ["ENCODED_MCT", "TA_YM", "Sales_Risk", "Customer_Risk", "Market_Risk", "RiskScore", "p_model", "p_final",
            "Alert"]
    return risks[cols]