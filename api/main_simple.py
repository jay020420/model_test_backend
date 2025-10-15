# api/main_simple.py - DB 없이 실행 가능한 간소화 버전
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="SME Early Warning API (Simple Mode)",
    version="2.0.0",
    description="소상공인 조기경보 시스템 - 간소화 버전 (DB 불필요)"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "service": "SME Early Warning API",
        "version": "2.0.0-simple",
        "status": "running",
        "mode": "simple (no database)",
        "docs": "/docs"
    }

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "ok",
        "version": "2.0.0-simple",
        "components": {
            "database": "disabled",
            "redis": "disabled",
            "model": "standalone"
        }
    }

# NLP 파싱 엔드포인트
@app.post("/api/v1/nlp/parse")
async def parse_utterance(request: dict):
    from api.service.nlp import parse_utterance
    
    utterance = request.get("utterance", "")
    result = parse_utterance(utterance)
    return result

# 즉시 위험도 스코어링
@app.post("/api/v1/predict/quickscore")
async def quick_score(request: dict):
    from api.service.prediction import quickscore, generate_recommendations
    
    result = quickscore(request)
    result["recommendations"] = generate_recommendations(result)
    
    import uuid
    from datetime import datetime
    result["id"] = str(uuid.uuid4())
    result["timestamp"] = datetime.utcnow().isoformat()
    
    return result

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Simple Mode Server...")
    print("📍 API Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)