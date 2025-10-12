# api/routes/analysis.py - 분석 API
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class BenchmarkRequest(BaseModel):
    industry_code: str
    region_code: Optional[str] = None
    metric: str = "p_final"


@router.post("/benchmark")
async def benchmark(request: BenchmarkRequest):
    """업종/지역 벤치마크"""
    from ..service.analysis import get_benchmark
    
    try:
        result = await get_benchmark(
            request.industry_code,
            request.region_code,
            request.metric
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))