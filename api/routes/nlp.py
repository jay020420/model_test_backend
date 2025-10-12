# api/routes/nlp.py - NLP 파싱
from fastapi import APIRouter, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address
from pydantic import BaseModel

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


class ParseRequest(BaseModel):
    utterance: str


class ParseResponse(BaseModel):
    industry_code: str = None
    region_code: str = None
    delivery_share: float = None
    sales_1m: float = None
    sales_3m_avg: float = None
    cust_1m: float = None
    cust_3m_avg: float = None


@router.post("/parse", response_model=ParseResponse)
@limiter.limit("30/minute")
async def parse_utterance_endpoint(request: ParseRequest):
    """자연어 파싱"""
    from ..service.nlp import parse_utterance
    
    try:
        result = parse_utterance(request.utterance)
        return ParseResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
