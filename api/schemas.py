from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List
from datetime import datetime
from enum import Enum


# ============ 기존 스키마 (강화) ============
class PredictRequest(BaseModel):
    store_id: Optional[str] = None
    target_month: Optional[str] = Field(None, description="YYYY-MM")
    region_code: Optional[str] = None
    industry_code: Optional[str] = None
    delivery_share: Optional[float] = Field(None, ge=0.0, le=1.0, description="0~1 (배달 위주=0.8)")
    sales_1m: Optional[float] = Field(None, ge=0, description="최근 1개월 매출(원)")
    sales_3m_avg: Optional[float] = Field(None, ge=0, description="최근 3개월 평균 매출(원)")
    cust_1m: Optional[float] = Field(None, ge=0, description="최근 1개월 고객수(명)")
    cust_3m_avg: Optional[float] = Field(None, ge=0, description="최근 3개월 평균 고객수(명)")

    @validator('target_month')
    def validate_month(cls, v):
        if v:
            try:
                datetime.strptime(v, '%Y-%m')
            except ValueError:
                raise ValueError('target_month must be in YYYY-MM format')
        return v


class RiskLevel(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED = "RED"


class PredictResponse(BaseModel):
    store_id: Optional[str]
    target_month: Optional[str]
    p_model: float
    risk_components: Dict[str, float]
    risk_score: float
    p_final: float
    alert: RiskLevel
    explanations: List[str]
    recommendations: Optional[List[str]] = None  # 추가: 개선 제안
    timestamp: datetime = Field(default_factory=datetime.now)


# ============ 챗봇 대화 스키마 ============
class ChatMessage(BaseModel):
    role: str = Field(..., description="user 또는 assistant")
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    context: Optional[Dict] = None  # 이전 대화 맥락


class ChatResponse(BaseModel):
    session_id: str
    message: str
    parsed_data: Optional[Dict] = None  # 파싱된 데이터
    prediction: Optional[PredictResponse] = None  # 예측 결과
    follow_up_questions: Optional[List[str]] = None  # 후속 질문
    needs_more_info: bool = False
    missing_fields: Optional[List[str]] = None


# ============ 비교 분석 스키마 ============
class BenchmarkRequest(BaseModel):
    industry_code: str
    region_code: Optional[str] = None
    metric: str = Field(default="p_final", description="비교할 지표")


class BenchmarkResponse(BaseModel):
    industry_code: str
    region_code: Optional[str]
    metric: str
    user_value: float
    industry_avg: float
    industry_median: float
    industry_percentile: float  # 사용자가 상위 몇%인지
    interpretation: str


# ============ 설명 API 스키마 ============
class ExplainRequest(BaseModel):
    component: str = Field(..., description="Sales_Risk, Customer_Risk, Market_Risk 중 하나")
    value: float


class ExplainResponse(BaseModel):
    component: str
    value: float
    severity: str  # Low, Medium, High, Critical
    factors: List[Dict[str, any]]  # 구성 요소별 상세
    actionable_insights: List[str]  # 실행 가능한 인사이트


# ============ 시계열 예측 스키마 ============
class ForecastRequest(BaseModel):
    store_id: str
    months_ahead: int = Field(default=3, ge=1, le=12)


class ForecastResponse(BaseModel):
    store_id: str
    forecasts: List[Dict[str, any]]  # [{month, p_final, confidence_lower, confidence_upper}]
    trend: str  # improving, stable, declining
    warnings: List[str]


# ============ 알림 설정 스키마 ============
class AlertConfig(BaseModel):
    store_id: str
    yellow_threshold: float = Field(default=0.20, ge=0, le=1)
    orange_threshold: float = Field(default=0.30, ge=0, le=1)
    red_threshold: float = Field(default=0.40, ge=0, le=1)
    notification_channels: List[str] = ["email"]  # email, sms, webhook


class AlertConfigResponse(BaseModel):
    store_id: str
    config: AlertConfig
    status: str
    message: str


# ============ 히스토리 스키마 ============
class HistoryRequest(BaseModel):
    store_id: Optional[str] = None
    start_date: Optional[str] = None  # YYYY-MM
    end_date: Optional[str] = None  # YYYY-MM
    limit: int = Field(default=100, ge=1, le=1000)


class HistoryResponse(BaseModel):
    records: List[PredictResponse]
    total_count: int
    has_more: bool


# ============ 피드백 스키마 ============
class FeedbackRequest(BaseModel):
    session_id: str
    prediction_id: Optional[str] = None
    rating: int = Field(..., ge=1, le=5, description="1-5 별점")
    comment: Optional[str] = None
    was_accurate: Optional[bool] = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    status: str
    message: str


# ============ 배치 분석 스키마 ============
class BatchAnalysisRequest(BaseModel):
    stores: List[PredictRequest]


class BatchAnalysisResponse(BaseModel):
    results: List[PredictResponse]
    summary: Dict[str, any]  # 전체 통계
    high_risk_stores: List[str]  # 위험 매장 ID 목록


# ============ 리포트 생성 스키마 ============
class ReportRequest(BaseModel):
    store_id: Optional[str] = None
    start_date: str  # YYYY-MM
    end_date: str  # YYYY-MM
    format: str = Field(default="json", description="json, pdf, excel")
    include_charts: bool = True


class ReportResponse(BaseModel):
    report_id: str
    download_url: Optional[str] = None
    content: Optional[Dict] = None  # JSON 형식인 경우
    generated_at: datetime = Field(default_factory=datetime.now)