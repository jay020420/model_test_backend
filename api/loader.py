import os, pandas as pd, joblib

BASE_DIR = os.environ.get("BASE_DIR", "/Users/llouis/Documents/model_test")
DATA_DIR = os.path.join(BASE_DIR, "data")
ART_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

CAL_PLATT_PATH = os.path.join(ART_DIR, "platt.joblib")
RISK_OUTPUT_PATH = os.path.join(BASE_DIR, "risk_output_trained.csv")


def load_risk_output():
    if os.path.exists(RISK_OUTPUT_PATH):
        df = pd.read_csv(RISK_OUTPUT_PATH)
        df["ENCODED_MCT"] = df["ENCODED_MCT"].astype(str)
        df["TA_YM"] = pd.to_datetime(df["TA_YM"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        return df
    return None


def load_platt():
    if os.path.exists(CAL_PLATT_PATH):
        return joblib.load(CAL_PLATT_PATH)
    return None