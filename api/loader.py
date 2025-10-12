import os
import pandas as pd

from .config import settings

def load_risk_output():
    """위험도 출력 파일 로드"""
    if os.path.exists(settings.RISK_OUTPUT_PATH):
        df = pd.read_csv(settings.RISK_OUTPUT_PATH)
        df["ENCODED_MCT"] = df["ENCODED_MCT"].astype(str)
        df["TA_YM"] = pd.to_datetime(df["TA_YM"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        return df
    return None