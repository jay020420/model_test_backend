# api/routes/admin.py - 관리자 API
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db, get_api_usage_stats

router = APIRouter()


@router.get("/stats")
async def get_stats(
    days: int = 7,
    db: AsyncSession = Depends(get_db)
):
    """API 사용 통계"""
    try:
        stats = await get_api_usage_stats(db, days=days)
        return {
            "period_days": days,
            "endpoints": [
                {
                    "endpoint": s.endpoint,
                    "count": s.count,
                    "avg_response_time": round(s.avg_response_time, 3)
                }
                for s in stats
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))