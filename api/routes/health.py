# api/routes/health.py - 헬스체크
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from ..database import get_db
from ..cache import get_cache

router = APIRouter()


@router.get("/health")
async def health_check(
    db: AsyncSession = Depends(get_db)
):
    """서비스 헬스체크"""
    from ..config import settings
    import os
    
    # DB 연결 확인
    db_status = "healthy"
    try:
        await db.execute("SELECT 1")
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Redis 연결 확인
    cache = await get_cache()
    redis_status = "healthy" if cache else "unavailable"
    
    # 모델 파일 확인
    model_status = "healthy" if os.path.exists(settings.RISK_OUTPUT_PATH) else "model file not found"
    
    return {
        "status": "ok" if db_status == "healthy" else "degraded",
        "version": "2.0.0",
        "environment": settings.ENVIRONMENT,
        "components": {
            "database": db_status,
            "redis": redis_status,
            "model": model_status
        }
    }


@router.get("/readiness")
async def readiness_check():
    """준비 상태 확인 (Kubernetes용)"""
    return {"ready": True}


@router.get("/liveness")
async def liveness_check():
    """활성 상태 확인 (Kubernetes용)"""
    return {"alive": True}