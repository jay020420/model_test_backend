import numpy as np
from typing import Dict, Optional, List
from .loader import load_risk_output

ALPHA, BETA, GAMMA = 0.4, 0.3, 0.3

BASE_MARKET = {
    "default": 0.55,
    "치킨": 0.56,
    "카페": 0.54,
    "피자": 0.57,
    "편의점": 0.53,
}


def _market_risk(industry: Optional[str], region: Optional[str], delivery_share: Optional[float]) -> float:
    base = BASE_MARKET.get(industry, BASE_MARKET["default"])
    if delivery_share is not None:
        base += 0.02 * (delivery_share - 0.5)
    if region and region.endswith("구"):
        base += 0.002
    return float(np.clip(base, 0.45, 0.65))


def _label_alert(p_final: float, green_q=0.7, orange_q=0.9) -> str:
    if p_final >= orange_q: return "RED"
    if p_final >= green_q: return "ORANGE"
    return "GREEN"


def _explain(rc: Dict[str, float]) -> List[str]:
    e = []
    if rc.get("Sales_Risk", 0) > 0.05: e.append("최근 1→3개월 대비 매출 모멘텀 둔화")
    if rc.get("Customer_Risk", 0) > 0.05: e.append("고객 수 감소 신호")
    if rc.get("Market_Risk", 0) > 0.55: e.append("지역/업종 시장 위험도 상회")
    return e or ["위험 신호는 크지 않음"]


def compute_rule_risks(sales_1m: Optional[float], sales_3m_avg: Optional[float],
                       cust_1m: Optional[float], cust_3m_avg: Optional[float],
                       industry_code: Optional[str], region_code: Optional[str],
                       delivery_share: Optional[float]) -> Dict[str, float]:
    sales_risk, cust_risk = 0.0, 0.0
    if sales_1m and sales_3m_avg and sales_3m_avg > 0:
        mom = (sales_1m - sales_3m_avg) / (sales_3m_avg + 1e-9)
        sales_risk = float(np.clip(-mom, 0, 1) * 0.1)
    if cust_1m and cust_3m_avg and cust_3m_avg > 0:
        cmom = (cust_1m - cust_3m_avg) / (cust_3m_avg + 1e-9)
        cust_risk = float(np.clip(-cmom, 0, 1) * 0.1)
    market_risk = _market_risk(industry_code, region_code, delivery_share)
    return {
        "Sales_Risk": round(sales_risk, 6),
        "Customer_Risk": round(cust_risk, 6),
        "Market_Risk": round(market_risk, 6),
    }


def blend_final(p_model: Optional[float], rc: Dict[str, float]) -> float:
    risk_score = ALPHA * rc["Sales_Risk"] + BETA * rc["Customer_Risk"] + GAMMA * rc["Market_Risk"]
    if p_model is None:
        return float(np.clip(risk_score, 0, 1))
    return float(np.clip(0.4 * risk_score + 0.6 * p_model, 0, 1))


def predict_batch(store_id: Optional[str], target_month: Optional[str]) -> Optional[Dict]:
    df = load_risk_output()
    if df is None:
        return None
    import pandas as pd
    tm = pd.to_datetime((target_month + "-01") if target_month else None, errors="coerce")
    sub = df.copy()
    if store_id: sub = sub[sub["ENCODED_MCT"] == str(store_id)]
    if tm is not None: sub = sub[sub["TA_YM"] == tm]
    if sub.empty: return None
    row = sub.sort_values("TA_YM").iloc[-1]
    rc = {
        "Sales_Risk": float(row.get("Sales_Risk", 0.0)),
        "Customer_Risk": float(row.get("Customer_Risk", 0.0)),
        "Market_Risk": float(row.get("Market_Risk", 0.0)),
    }
    p_model = float(row.get("p_model", 0.0))
    p_final = float(row.get("p_final", 0.0)) if "p_final" in sub.columns else blend_final(p_model, rc)
    return {
        "store_id": store_id,
        "target_month": row["TA_YM"].strftime("%Y-%m"),
        "p_model": p_model,
        "risk_components": rc,
        "risk_score": round(ALPHA * rc["Sales_Risk"] + BETA * rc["Customer_Risk"] + GAMMA * rc["Market_Risk"], 6),
        "p_final": p_final,
        "alert": _label_alert(p_final),
        "explanations": _explain(rc)
    }


def quickscore(payload: Dict) -> Dict:
    rc = compute_rule_risks(
        payload.get("sales_1m"),
        payload.get("sales_3m_avg"),
        payload.get("cust_1m"),
        payload.get("cust_3m_avg"),
        payload.get("industry_code"),
        payload.get("region_code"),
        payload.get("delivery_share"),
    )
    p_model = None
    p_final = blend_final(p_model, rc)
    return {
        "store_id": payload.get("store_id"),
        "target_month": payload.get("target_month"),
        "p_model": 0.0,
        "risk_components": rc,
        "risk_score": round(ALPHA * rc["Sales_Risk"] + BETA * rc["Customer_Risk"] + GAMMA * rc["Market_Risk"], 6),
        "p_final": float(p_final),
        "alert": _label_alert(p_final),
        "explanations": _explain(rc)
    }