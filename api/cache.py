# ===== api/cache.py - Redis 캐시 설정 =====
import redis.asyncio as redis
from typing import Optional
import json

from .config import settings

# Redis 클라이언트
_redis_client: Optional[redis.Redis] = None


async def init_cache():
    """Redis 초기화"""
    global _redis_client
    
    try:
        _redis_client = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5
        )
        # 연결 테스트
        await _redis_client.ping()
        print("Redis connected successfully")
    except Exception as e:
        print(f"Redis connection failed: {e}")
        _redis_client = None


async def close_cache():
    """Redis 연결 종료"""
    global _redis_client
    if _redis_client:
        await _redis_client.close()


async def get_cache() -> Optional[redis.Redis]:
    """Redis 클라이언트 반환"""
    return _redis_client


# ===== 캐시 유틸리티 함수 =====
async def cache_get(key: str):
    """캐시에서 데이터 가져오기"""
    if not _redis_client:
        return None
    
    try:
        value = await _redis_client.get(key)
        if value:
            return json.loads(value)
    except Exception as e:
        print(f"Cache get error: {e}")
    
    return None


async def cache_set(key: str, value: any, ttl: int = None):
    """캐시에 데이터 저장"""
    if not _redis_client:
        return False
    
    try:
        ttl = ttl or settings.CACHE_TTL
        await _redis_client.setex(
            key,
            ttl,
            json.dumps(value, default=str)
        )
        return True
    except Exception as e:
        print(f"Cache set error: {e}")
        return False


async def cache_delete(key: str):
    """캐시에서 데이터 삭제"""
    if not _redis_client:
        return False
    
    try:
        await _redis_client.delete(key)
        return True
    except Exception as e:
        print(f"Cache delete error: {e}")
        return False


async def cache_exists(key: str) -> bool:
    """캐시 키 존재 여부 확인"""
    if not _redis_client:
        return False
    
    try:
        return await _redis_client.exists(key) > 0
    except Exception:
        return False


# 세션 관리
async def save_session(session_id: str, data: dict, ttl: int = None):
    """세션 저장"""
    key = f"session:{session_id}"
    ttl = ttl or settings.SESSION_TTL
    return await cache_set(key, data, ttl)


async def get_session(session_id: str):
    """세션 조회"""
    key = f"session:{session_id}"
    return await cache_get(key)


async def delete_session(session_id: str):
    """세션 삭제"""
    key = f"session:{session_id}"
    return await cache_delete(key)