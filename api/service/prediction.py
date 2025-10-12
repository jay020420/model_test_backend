# api/service/prediction.py (기존 service.py 기반)
import numpy as np
from typing import Dict, Optional, List

ALPHA, BETA, GAMMA = 0.4, 0.3, 0.3

BASE_MARKET = {
    "default": 0.55,
    "치킨": 0.56,
    "카페": 0.54,
    "피자": 0.57,
    "편의점": 0.53,
}


def _market_risk(industry: Optional[str], region: Optional[str], delivery_share: Optional[float]) -> float:
    """시장 위험도 계산"""
    base = BASE_MARKET.get(industry, BASE_MARKET["default"])
    
    if delivery_share is not None:
        base += 0.02 * (delivery_share - 0.5)
    
    if region and region.endswith("구"):
        base += 0.002
    
    return float(np.clip(base, 0.45, 0.65))


def compute_rule_risks(
    sales_1m: Optional[float],
    sales_3m_avg: Optional[float],
    cust_1m: Optional[float],
    cust_3m_avg: Optional[float],
    industry_code: Optional[str],
    region_code: Optional[str],
    delivery_share: Optional[float]
) -> Dict[str, float]:
    """규칙 기반 위험도 계산"""
    sales_risk, cust_risk = 0.0, 0.0
    
    # 매출 위험도
    if sales_1m and sales_3m_avg and sales_3m_avg > 0:
        mom = (sales_1m - sales_3m_avg) / (sales_3m_avg + 1e-9)
        sales_risk = float(np.clip(-mom, 0, 1) * 0.1)
    
    # 고객 위험도
    if cust_1m and cust_3m_avg and cust_3m_avg > 0:
        cmom = (cust_1m - cust_3m_avg) / (cust_3m_avg + 1e-9)
        cust_risk = float(np.clip(-cmom, 0, 1) * 0.1)
    
    # 시장 위험도
    market_risk = _market_risk(industry_code, region_code, delivery_share)
    
    return {
        "Sales_Risk": round(sales_risk, 6),
        "Customer_Risk": round(cust_risk, 6),
        "Market_Risk": round(market_risk, 6),
    }


def _label_alert(p_final: float, green_q=0.7, orange_q=0.9) -> str:
    """경보 레벨 결정"""
    if p_final >= orange_q:
        return "RED"
    if p_final >= green_q:
        return "ORANGE"
    return "GREEN"


def _explain(rc: Dict[str, float]) -> List[str]:
    """설명 생성"""
    e = []
    if rc.get("Sales_Risk", 0) > 0.05:
        e.append("최근 1→3개월 대비 매출 모멘텀 둔화")
    if rc.get("Customer_Risk", 0) > 0.05:
        e.append("고객 수 감소 신호")
    if rc.get("Market_Risk", 0) > 0.55:
        e.append("지역/업종 시장 위험도 상회")
    return e or ["위험 신호는 크지 않음"]


def quickscore(payload: Dict) -> Dict:
    """즉시 스코어링"""
    rc = compute_rule_risks(
        payload.get("sales_1m"),
        payload.get("sales_3m_avg"),
        payload.get("cust_1m"),
        payload.get("cust_3m_avg"),
        payload.get("industry_code"),
        payload.get("region_code"),
        payload.get("delivery_share"),
    )
    
    risk_score = ALPHA * rc["Sales_Risk"] + BETA * rc["Customer_Risk"] + GAMMA * rc["Market_Risk"]
    p_final = float(np.clip(risk_score, 0, 1))
    
    return {
        "store_id": payload.get("store_id"),
        "target_month": payload.get("target_month"),
        "p_model": 0.0,
        "risk_components": rc,
        "risk_score": round(risk_score, 6),
        "p_final": p_final,
        "alert": _label_alert(p_final),
        "explanations": _explain(rc)
    }


def predict_batch(store_id: Optional[str], target_month: Optional[str]) -> Optional[Dict]:
    """배치 예측 (학습된 모델 사용)"""
    from ..loader import load_risk_output
    import pandas as pd
    
    df = load_risk_output()
    if df is None:
        return None
    
    tm = pd.to_datetime((target_month + "-01") if target_month else None, errors="coerce")
    
    sub = df.copy()
    if store_id:
        sub = sub[sub["ENCODED_MCT"] == str(store_id)]
    if tm is not None:
        sub = sub[sub["TA_YM"] == tm]
    
    if sub.empty:
        return None
    
    row = sub.sort_values("TA_YM").iloc[-1]
    
    rc = {
        "Sales_Risk": float(row.get("Sales_Risk", 0.0)),
        "Customer_Risk": float(row.get("Customer_Risk", 0.0)),
        "Market_Risk": float(row.get("Market_Risk", 0.0)),
    }
    
    p_model = float(row.get("p_model", 0.0))
    p_final = float(row.get("p_final", 0.0))
    
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


def generate_recommendations(result: Dict) -> List[str]:
    """개선 제안 생성"""
    recommendations = []
    
    rc = result.get("risk_components", {})
    
    if rc.get("Sales_Risk", 0) > 0.05:
        recommendations.extend([
            "💰 매출 개선: 프로모션 이벤트나 신메뉴 출시를 고려하세요",
            "📊 가격 전략: 경쟁사 가격을 분석하고 차별화된 가격 정책을 수립하세요"
        ])
    
    if rc.get("Customer_Risk", 0) > 0.05:
        recommendations.extend([
            "👥 고객 유지: 단골 고객 대상 리워드 프로그램을 운영하세요",
            "⭐ 리뷰 관리: 부정적 리뷰에 신속히 대응하고 개선하세요"
        ])
    
    if rc.get("Market_Risk", 0) > 0.55:
        recommendations.extend([
            "🎯 차별화: 경쟁업체와 차별화된 강점을 부각하세요",
            "📱 온라인 강화: 배달앱 외 자체 채널을 개발하세요"
        ])
    
    if result.get("alert") == "RED":
        recommendations.append("⚠️ 긴급 조치: 비용 구조 재검토 및 전문가 상담을 권장합니다")
    
    return recommendations[:5] or ["✅ 현재 위험 신호는 크지 않습니다. 지속적으로 모니터링하세요."]