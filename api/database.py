# api/database.py - PostgreSQL 설정
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Float, DateTime, Integer, Text, JSON
from datetime import datetime

from .config import settings

# SQLAlchemy 엔진 생성
engine = create_async_engine(
    settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
    echo=settings.DEBUG,
    pool_size=10,
    max_overflow=20
)

# 세션 팩토리
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base 모델
Base = declarative_base()


# ===== 모델 정의 =====
class PredictionHistory(Base):
    """예측 이력"""
    __tablename__ = "prediction_history"
    
    id = Column(String, primary_key=True)
    session_id = Column(String, index=True)
    store_id = Column(String, index=True, nullable=True)
    target_month = Column(String)
    
    # 입력 데이터
    industry_code = Column(String)
    region_code = Column(String)
    delivery_share = Column(Float)
    sales_1m = Column(Float)
    sales_3m_avg = Column(Float)
    cust_1m = Column(Float)
    cust_3m_avg = Column(Float)
    
    # 결과
    p_model = Column(Float)
    p_final = Column(Float)
    risk_score = Column(Float)
    alert = Column(String)
    risk_components = Column(JSON)
    
    # 메타데이터
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    api_key = Column(String, index=True)
    ip_address = Column(String)
    user_agent = Column(String)


class ChatSession(Base):
    """채팅 세션"""
    __tablename__ = "chat_sessions"
    
    session_id = Column(String, primary_key=True)
    messages = Column(JSON)  # 메시지 배열
    parsed_data = Column(JSON)  # 파싱된 데이터
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    api_key = Column(String, index=True)
    completed = Column(Integer, default=0)  # 0: 진행중, 1: 완료


class Feedback(Base):
    """사용자 피드백"""
    __tablename__ = "feedbacks"
    
    id = Column(String, primary_key=True)
    session_id = Column(String, index=True)
    prediction_id = Column(String, index=True, nullable=True)
    
    rating = Column(Integer)  # 1-5
    comment = Column(Text)
    was_accurate = Column(Integer)  # 0: false, 1: true, NULL: unknown
    
    created_at = Column(DateTime, default=datetime.utcnow)
    api_key = Column(String)


class AlertConfig(Base):
    """알림 설정"""
    __tablename__ = "alert_configs"
    
    store_id = Column(String, primary_key=True)
    
    yellow_threshold = Column(Float, default=0.20)
    orange_threshold = Column(Float, default=0.30)
    red_threshold = Column(Float, default=0.40)
    
    notification_channels = Column(JSON)  # ["email", "sms", "webhook"]
    email = Column(String)
    phone = Column(String)
    webhook_url = Column(String)
    
    enabled = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class APIUsage(Base):
    """API 사용 통계"""
    __tablename__ = "api_usage"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    api_key = Column(String, index=True)
    endpoint = Column(String, index=True)
    method = Column(String)
    
    status_code = Column(Integer)
    response_time = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


# ===== 데이터베이스 초기화 =====
async def init_db():
    """데이터베이스 초기화"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """데이터베이스 연결 종료"""
    await engine.dispose()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """데이터베이스 세션 의존성"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# ===== 리포지토리 함수 =====
async def save_prediction(db: AsyncSession, prediction: dict):
    """예측 결과 저장"""
    history = PredictionHistory(**prediction)
    db.add(history)
    await db.commit()
    return history


async def get_prediction_history(
    db: AsyncSession,
    store_id: str = None,
    limit: int = 100
):
    """예측 이력 조회"""
    from sqlalchemy import select, desc
    
    query = select(PredictionHistory).order_by(desc(PredictionHistory.created_at))
    
    if store_id:
        query = query.where(PredictionHistory.store_id == store_id)
    
    query = query.limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()


async def save_feedback(db: AsyncSession, feedback: dict):
    """피드백 저장"""
    fb = Feedback(**feedback)
    db.add(fb)
    await db.commit()
    return fb


async def get_api_usage_stats(
    db: AsyncSession,
    api_key: str = None,
    days: int = 7
):
    """API 사용 통계"""
    from sqlalchemy import select, func
    from datetime import timedelta
    
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    query = select(
        APIUsage.endpoint,
        func.count(APIUsage.id).label('count'),
        func.avg(APIUsage.response_time).label('avg_response_time')
    ).where(
        APIUsage.created_at >= cutoff
    )
    
    if api_key:
        query = query.where(APIUsage.api_key == api_key)
    
    query = query.group_by(APIUsage.endpoint)
    
    result = await db.execute(query)
    return result.all()