from pydantic import BaseModel, Field
from typing import Optional, Dict, List


class PredictRequest(BaseModel):
    store_id: Optional[str] = None
    target_month: Optional[str] = Field(None, description="YYYY-MM")
    region_code: Optional[str] = None
    industry_code: Optional[str] = None
    delivery_share: Optional[float] = Field(None, description="0~1 (배달 위주=0.8 등)")
    sales_1m: Optional[float] = None
    sales_3m_avg: Optional[float] = None
    cust_1m: Optional[float] = None
    cust_3m_avg: Optional[float] = None


class PredictResponse(BaseModel):
    store_id: Optional[str]
    target_month: Optional[str]
    p_model: float
    risk_components: Dict[str, float]
    risk_score: float
    p_final: float
    alert: str
    explanations: List[str]


class ParseRequest(BaseModel):
    utterance: str


class ParseResponse(BaseModel):
    industry_code: Optional[str]
    region_code: Optional[str]
    delivery_share: Optional[float]
    sales_1m: Optional[float]
    sales_3m_avg: Optional[float]
    cust_1m: Optional[float]
    cust_3m_avg: Optional[float]