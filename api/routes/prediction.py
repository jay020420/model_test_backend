# api/routes/prediction.py - 예측 API
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
import uuid

from ..database import get_db, save_prediction
from ..cache import cache_get, cache_set

router = APIRouter()


class PredictRequest(BaseModel):
    store_id: Optional[str] = None
    target_month: Optional[str] = Field(None, description="YYYY-MM")
    region_code: Optional[str] = None
    industry_code: Optional[str] = None
    delivery_share: Optional[float] = Field(None, ge=0.0, le=1.0)
    sales_1m: Optional[float] = Field(None, ge=0)
    sales_3m_avg: Optional[float] = Field(None, ge=0)
    cust_1m: Optional[float] = Field(None, ge=0)
    cust_3m_avg: Optional[float] = Field(None, ge=0)


class PredictResponse(BaseModel):
    id: str
    store_id: Optional[str]
    target_month: Optional[str]
    p_model: float
    risk_components: Dict[str, float]
    risk_score: float
    p_final: float
    alert: str
    explanations: List[str]
    recommendations: List[str]
    timestamp: datetime


@router.post("/quickscore", response_model=PredictResponse)
async def quick_score(
    request: PredictRequest,
    req: Request,
    db: AsyncSession = Depends(get_db)
):
    """즉시 위험도 스코어링"""
    from ..service.prediction import quickscore, generate_recommendations
    
    try:
        # 캐시 키 생성 (동일한 입력에 대해 캐싱)
        cache_key = f"quickscore:{hash(str(request.dict()))}"
        cached = await cache_get(cache_key)
        
        if cached:
            return PredictResponse(**cached)
        
        # 스코어링
        result = quickscore(request.dict())
        result["recommendations"] = generate_recommendations(result)
        
        # ID 및 타임스탬프 추가
        prediction_id = str(uuid.uuid4())
        result["id"] = prediction_id
        result["timestamp"] = datetime.utcnow()
        
        # 캐시 저장 (5분)
        await cache_set(cache_key, result, ttl=300)
        
        # DB 저장 (비동기)
        await save_prediction(db, {
            "id": prediction_id,
            **request.dict(),
            **result,
            "api_key": req.state.api_key_info.get("name") if hasattr(req.state, "api_key_info") else None,
            "ip_address": req.client.host if req.client else None,
            "user_agent": req.headers.get("user-agent")
        })
        
        return PredictResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 실패: {str(e)}")


@router.post("/model", response_model=PredictResponse)
async def model_predict(
    request: PredictRequest,
    req: Request,
    db: AsyncSession = Depends(get_db)
):
    """ML 모델 기반 예측"""
    from ..service.prediction import predict_batch, quickscore, generate_recommendations
    
    try:
        # 모델 예측 시도
        result = predict_batch(request.store_id, request.target_month)
        
        # 실패 시 quickscore로 폴백
        if result is None:
            if any([request.sales_1m, request.sales_3m_avg, request.cust_1m, request.cust_3m_avg]):
                result = quickscore(request.dict())
            else:
                raise HTTPException(status_code=404, detail="매장 데이터를 찾을 수 없습니다.")
        
        result["recommendations"] = generate_recommendations(result)
        
        # ID 및 타임스탬프 추가
        prediction_id = str(uuid.uuid4())
        result["id"] = prediction_id
        result["timestamp"] = datetime.utcnow()
        
        # DB 저장
        await save_prediction(db, {
            "id": prediction_id,
            **request.dict(),
            **result,
            "api_key": req.state.api_key_info.get("name") if hasattr(req.state, "api_key_info") else None,
            "ip_address": req.client.host if req.client else None
        })
        
        return PredictResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 실패: {str(e)}")


@router.get("/history/{store_id}")
async def get_store_history(
    store_id: str,
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """매장별 예측 이력"""
    from ..database import get_prediction_history
    
    try:
        history = await get_prediction_history(db, store_id=store_id, limit=limit)
        return {
            "store_id": store_id,
            "count": len(history),
            "predictions": [
                {
                    "id": h.id,
                    "target_month": h.target_month,
                    "p_final": h.p_final,
                    "alert": h.alert,
                    "created_at": h.created_at
                }
                for h in history
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))