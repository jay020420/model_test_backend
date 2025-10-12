# api/routes/chat.py - 챗봇
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import uuid

from ..cache import save_session, get_session, delete_session

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    context: Optional[Dict] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatResponse(BaseModel):
    session_id: str
    message: str
    parsed_data: Optional[Dict] = None
    prediction: Optional[Dict] = None
    follow_up_questions: Optional[List[str]] = None
    needs_more_info: bool = False
    missing_fields: Optional[List[str]] = None


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """챗봇 대화"""
    from ..service.nlp import parse_utterance
    from ..service.prediction import quickscore, generate_recommendations
    
    # 세션 관리
    session_id = request.session_id or str(uuid.uuid4())
    session = await get_session(session_id) or {
        "messages": [],
        "parsed_data": {}
    }
    
    # 메시지 저장
    session["messages"].append({"role": "user", "content": request.message})
    
    # NLP 파싱
    parsed = parse_utterance(request.message)
    session["parsed_data"].update({k: v for k, v in parsed.items() if v is not None})
    
    # 필수 필드 확인
    required = ["industry_code", "region_code", "sales_1m", "sales_3m_avg"]
    missing = [f for f in required if f not in session["parsed_data"]]
    
    if missing:
        # 정보 부족
        follow_up = generate_follow_up_questions(missing)
        response_msg = f"정보를 더 알려주세요: {', '.join(follow_up)}"
        
        session["messages"].append({"role": "assistant", "content": response_msg})
        await save_session(session_id, session)
        
        return ChatResponse(
            session_id=session_id,
            message=response_msg,
            parsed_data=session["parsed_data"],
            needs_more_info=True,
            missing_fields=missing,
            follow_up_questions=follow_up
        )
    
    # 예측 수행
    try:
        prediction = quickscore(session["parsed_data"])
        prediction["recommendations"] = generate_recommendations(prediction)
        
        # 응답 메시지 생성
        response_msg = generate_chat_response(prediction, session["parsed_data"])
        session["messages"].append({"role": "assistant", "content": response_msg})
        
        await save_session(session_id, session)
        
        return ChatResponse(
            session_id=session_id,
            message=response_msg,
            parsed_data=session["parsed_data"],
            prediction=prediction,
            needs_more_info=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 실패: {str(e)}")


@router.delete("/{session_id}")
async def clear_chat_session(session_id: str):
    """세션 삭제"""
    await delete_session(session_id)
    return {"status": "success", "message": "세션이 삭제되었습니다"}


def generate_follow_up_questions(missing: List[str]) -> List[str]:
    """부족한 정보에 대한 질문 생성"""
    questions = {
        "industry_code": "어떤 업종인가요?",
        "region_code": "어느 지역인가요?",
        "sales_1m": "최근 한 달 매출은?",
        "sales_3m_avg": "최근 3개월 평균 매출은?"
    }
    return [questions.get(f, f) for f in missing[:3]]


def generate_chat_response(prediction: Dict, data: Dict) -> str:
    """챗봇 응답 생성"""
    emoji = {"GREEN": "🟢", "YELLOW": "🟡", "ORANGE": "🟠", "RED": "🔴"}
    
    return f"""
{emoji.get(prediction['alert'], '⚪')} **위험도: {prediction['p_final']:.1%}** (경보: {prediction['alert']})

📊 세부 위험도:
- 매출: {prediction['risk_components']['Sales_Risk']:.1%}
- 고객: {prediction['risk_components']['Customer_Risk']:.1%}
- 시장: {prediction['risk_components']['Market_Risk']:.1%}

💡 주요 인사이트:
{chr(10).join(f"• {exp}" for exp in prediction['explanations'][:3])}

🎯 개선 제안:
{chr(10).join(f"• {rec}" for rec in prediction.get('recommendations', [])[:2])}
""".strip()