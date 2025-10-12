# api/main.py - 프로덕션 FastAPI 메인 애플리케이션
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

from .config import settings
from .routes import (
    nlp_router,
    prediction_router,
    chat_router,
    analysis_router,
    admin_router,
    health_router
)
from .middleware import (
    RequestLoggingMiddleware,
    TimingMiddleware,
    AuthenticationMiddleware
)
from .database import init_db, close_db
from .cache import init_cache, close_cache

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sentry 초기화 (에러 추적)
if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        integrations=[FastApiIntegration()],
        traces_sample_rate=0.1,
        environment=settings.ENVIRONMENT
    )

# Rate Limiter 설정
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # 시작 시
    logger.info("Starting SME Early Warning API...")
    await init_db()
    await init_cache()
    logger.info("Database and cache initialized")
    
    yield
    
    # 종료 시
    logger.info("Shutting down SME Early Warning API...")
    await close_db()
    await close_cache()
    logger.info("Cleanup completed")


# FastAPI 앱 생성
app = FastAPI(
    title="SME Early Warning API",
    version="2.0.0",
    description="소상공인 조기경보 시스템 - 프로덕션 API",
    docs_url="/api/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/api/redoc" if settings.ENVIRONMENT != "production" else None,
    lifespan=lifespan
)

# Rate Limiter 등록
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Prometheus 메트릭 (모니터링)
if settings.ENABLE_METRICS:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time"]
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(TimingMiddleware)
app.add_middleware(RequestLoggingMiddleware)

if settings.ENABLE_AUTH:
    app.add_middleware(AuthenticationMiddleware)

if settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )


# 전역 예외 처리
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """유효성 검증 실패 핸들러"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.errors(),
            "body": exc.body
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """전역 예외 핸들러"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    if settings.ENVIRONMENT == "production":
        # 프로덕션에서는 상세 에러 숨김
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": str(exc),
                "type": type(exc).__name__
            }
        )


# 라우터 등록
app.include_router(health_router, prefix="/api/v1", tags=["Health"])
app.include_router(nlp_router, prefix="/api/v1/nlp", tags=["NLP"])
app.include_router(prediction_router, prefix="/api/v1/predict", tags=["Prediction"])
app.include_router(chat_router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["Analysis"])
app.include_router(admin_router, prefix="/api/v1/admin", tags=["Admin"])


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "SME Early Warning API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/api/docs" if settings.ENVIRONMENT != "production" else None
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        workers=settings.WORKERS if settings.ENVIRONMENT == "production" else 1,
        log_level="info"
    )