# api/middleware.py - 커스텀 미들웨어
import time
import uuid
import logging
from typing import Callable
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .config import settings, validate_api_key

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """요청 로깅 미들웨어"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 요청 ID 생성
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # 요청 정보 로깅
        logger.info(
            f"Request {request_id}: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # 요청 처리
        response = await call_next(request)
        
        # 응답에 요청 ID 추가
        response.headers["X-Request-ID"] = request_id
        
        # 응답 로깅
        logger.info(
            f"Response {request_id}: status={response.status_code}"
        )
        
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """요청 처리 시간 측정 미들웨어"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(round(process_time, 3))
        
        # 느린 요청 경고
        if process_time > 1.0:
            logger.warning(
                f"Slow request: {request.method} {request.url.path} "
                f"took {process_time:.2f}s"
            )
        
        return response


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """API 키 인증 미들웨어"""
    
    # 인증이 필요 없는 경로
    PUBLIC_PATHS = [
        "/",
        "/api/docs",
        "/api/redoc",
        "/api/openapi.json",
        "/api/v1/health",
        "/metrics"
    ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 공개 경로는 인증 스킵
        if any(request.url.path.startswith(path) for path in self.PUBLIC_PATHS):
            return await call_next(request)
        
        # API 키 확인
        api_key = request.headers.get(settings.API_KEY_HEADER)
        
        if not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "API key required"}
            )
        
        if not validate_api_key(api_key):
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Invalid API key"}
            )
        
        # 요청 상태에 API 키 정보 저장
        from .config import get_api_key_info
        request.state.api_key_info = get_api_key_info(api_key)
        
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate Limiting 미들웨어 (Redis 기반)"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        from .cache import get_cache
        
        cache = await get_cache()
        if not cache:
            # 캐시 사용 불가 시 제한 없이 통과
            return await call_next(request)
        
        # 클라이언트 식별 (API 키 또는 IP)
        api_key = request.headers.get(settings.API_KEY_HEADER)
        client_id = api_key if api_key else (
            request.client.host if request.client else "unknown"
        )
        
        # Rate limit 키
        minute_key = f"ratelimit:{client_id}:minute"
        hour_key = f"ratelimit:{client_id}:hour"
        
        # 현재 카운트 확인
        minute_count = await cache.get(minute_key) or 0
        hour_count = await cache.get(hour_key) or 0
        
        # API 키별 제한 확인
        if hasattr(request.state, 'api_key_info'):
            limit = request.state.api_key_info.get('rate_limit', settings.RATE_LIMIT_PER_MINUTE)
        else:
            limit = settings.RATE_LIMIT_PER_MINUTE
        
        # 제한 확인
        if int(minute_count) >= limit:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        if int(hour_count) >= settings.RATE_LIMIT_PER_HOUR:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Hourly rate limit exceeded",
                    "retry_after": 3600
                },
                headers={"Retry-After": "3600"}
            )
        
        # 카운터 증가
        await cache.incr(minute_key)
        await cache.expire(minute_key, 60)
        await cache.incr(hour_key)
        await cache.expire(hour_key, 3600)
        
        response = await call_next(request)
        
        # Rate limit 헤더 추가
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, limit - int(minute_count) - 1))
        
        return response


class CacheMiddleware(BaseHTTPMiddleware):
    """응답 캐싱 미들웨어 (GET 요청만)"""
    
    CACHEABLE_PATHS = [
        "/api/v1/predict/benchmark",
        "/api/v1/analysis/explain"
    ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # GET 요청이 아니거나 캐시 가능한 경로가 아니면 스킵
        if request.method != "GET" or not any(
            request.url.path.startswith(path) for path in self.CACHEABLE_PATHS
        ):
            return await call_next(request)
        
        from .cache import get_cache
        import json
        
        cache = await get_cache()
        if not cache:
            return await call_next(request)
        
        # 캐시 키 생성
        cache_key = f"response:{request.url.path}:{request.url.query}"
        
        # 캐시 확인
        cached = await cache.get(cache_key)
        if cached:
            logger.info(f"Cache hit: {cache_key}")
            return JSONResponse(
                content=json.loads(cached),
                headers={"X-Cache": "HIT"}
            )
        
        # 요청 처리
        response = await call_next(request)
        
        # 성공 응답만 캐싱
        if response.status_code == 200:
            # 응답 본문 읽기
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            # 캐시 저장
            await cache.setex(
                cache_key,
                settings.CACHE_TTL,
                body.decode()
            )
            
            # 새 응답 생성
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers) | {"X-Cache": "MISS"},
                media_type=response.media_type
            )
        
        return response