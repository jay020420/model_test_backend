# api/config.py - 환경 설정
import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # 기본 설정
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    
    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = int(os.getenv("WORKERS", "4"))
    
    # CORS 설정
    CORS_ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:8080"
    ).split(",")
    
    ALLOWED_HOSTS: List[str] = os.getenv(
        "ALLOWED_HOSTS",
        "localhost,127.0.0.1"
    ).split(",")
    
    # 데이터베이스 설정
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://user:password@localhost:5432/sme_warning"
    )
    
    # Redis 설정 (캐시 & 세션)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # 1시간
    SESSION_TTL: int = int(os.getenv("SESSION_TTL", "86400"))  # 24시간
    
    # 모델 경로
    BASE_DIR: str = os.getenv("BASE_DIR", "/app")
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    ARTIFACTS_DIR: str = os.path.join(BASE_DIR, "artifacts")
    RISK_OUTPUT_PATH: str = os.path.join(BASE_DIR, "risk_output_trained.csv")
    
    # 인증 설정
    ENABLE_AUTH: bool = os.getenv("ENABLE_AUTH", "True").lower() == "true"
    API_KEY_HEADER: str = "X-API-Key"
    JWT_SECRET: str = os.getenv("JWT_SECRET", "jwt-secret-key-change-me")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    RATE_LIMIT_PER_HOUR: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))
    
    # 로깅
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Sentry (에러 추적)
    SENTRY_DSN: str = os.getenv("SENTRY_DSN", "")
    
    # 메트릭
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "True").lower() == "true"
    
    # 알림 설정
    EMAIL_HOST: str = os.getenv("EMAIL_HOST", "smtp.gmail.com")
    EMAIL_PORT: int = int(os.getenv("EMAIL_PORT", "587"))
    EMAIL_USER: str = os.getenv("EMAIL_USER", "")
    EMAIL_PASSWORD: str = os.getenv("EMAIL_PASSWORD", "")
    
    SMS_API_KEY: str = os.getenv("SMS_API_KEY", "")
    SMS_API_URL: str = os.getenv("SMS_API_URL", "")
    
    # 모델 설정
    MODEL_CACHE_SIZE: int = 100
    PREDICTION_TIMEOUT: int = 30  # 초
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()


# API 키 관리 (간단한 인메모리 저장소, 실제로는 DB 사용)
API_KEYS = {
    "dev_key_12345": {
        "name": "Development Key",
        "tier": "free",
        "rate_limit": 60
    },
    "prod_key_67890": {
        "name": "Production Key",
        "tier": "premium",
        "rate_limit": 1000
    }
}


def get_api_key_info(api_key: str) -> dict:
    """API 키 정보 조회"""
    return API_KEYS.get(api_key)


def validate_api_key(api_key: str) -> bool:
    """API 키 유효성 검증"""
    return api_key in API_KEYS