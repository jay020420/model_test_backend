# 🚨 SME Early Warning System (소상공인 조기경보 시스템)

머신러닝과 규칙 기반 위험도 분석을 통한 소상공인 경영 위험 조기 탐지 시스템

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 📋 목차

- [주요 기능](#주요-기능)
- [시스템 구조](#시스템-구조)
- [빠른 시작](#빠른-시작)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [API 문서](#api-문서)
- [배포](#배포)
- [개발](#개발)

---

## 🎯 주요 기능

### 1. 다차원 위험도 분석
- **매출 위험도**: 월별 매출 변동, AOV(객단가), 취소율 분석
- **고객 위험도**: 고객 수 변화, 충성도, 신규 고객 유입
- **시장 위험도**: 업종/지역 폐업률, 경쟁 환경 분석

### 2. 앙상블 예측 모델
- XGBoost, LightGBM, Random Forest, Gradient Boosting
- TensorFlow 딥러닝 모델
- Platt Scaling 확률 보정

### 3. 실시간 경보 시스템
- 🟢 **GREEN**: 정상 (위험도 < 20%)
- 🟡 **YELLOW**: 주의 (20% ≤ 위험도 < 30%)
- 🟠 **ORANGE**: 경고 (30% ≤ 위험도 < 40%)
- 🔴 **RED**: 위험 (위험도 ≥ 40%)

### 4. 프로덕션 API
- FastAPI 기반 RESTful API
- PostgreSQL + Redis 캐싱
- JWT 인증, Rate Limiting
- Prometheus 메트릭, Sentry 에러 추적

---

## 🏗️ 시스템 구조

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Client    │─────▶│  FastAPI     │─────▶│ PostgreSQL  │
│  (Web/App)  │      │   (API)      │      │   (DB)      │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                            ├─────▶ Redis (Cache)
                            │
                            └─────▶ ML Models (Ensemble)
```

---

## ⚡ 빠른 시작

### 필수 요구사항
- Python 3.11+
- Docker & Docker Compose (선택)
- PostgreSQL 15+ (Docker 사용 시 불필요)

### 1️⃣ 로컬 실행 (기본 위험도 분석)

```bash
# 1. 리포지토리 클론
git clone <repository-url>
cd model_test

# 2. 가상환경 생성 및 패키지 설치
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. 데이터 파일 준비 (data/ 디렉토리에 배치)
# - big_data_set1_f.csv
# - ds2_monthly_usage.csv
# - ds3_monthly_customers.csv

# 4. 위험도 분석 실행
python run.py

# 출력: risk_output.csv (위험도 결과)
```

### 2️⃣ API 서버 실행 (Docker)

```bash
# 1. 환경 변수 설정
cp .env.example .env
# .env 파일 편집 (DB 비밀번호, SECRET_KEY 등)

# 2. Docker 컨테이너 실행
docker-compose up -d

# 3. API 접근
# - API 문서: http://localhost:8000/api/docs
# - Health Check: http://localhost:8000/api/v1/health
```

---

## 📦 설치 방법

### Option A: 로컬 개발 환경

```bash
# Python 의존성 설치
pip install -r requirements.txt

# 개발용 추가 패키지 (선택)
pip install pytest black flake8 mypy
```

### Option B: Docker 환경

```bash
# 개발 환경 (hot-reload)
docker-compose -f docker-compose.dev.yml up

# 프로덕션 환경
docker-compose up -d
```

---

## 🚀 사용 방법

### 1. CLI: 배치 위험도 분석

```bash
# 기본 실행
python run.py

# 커스텀 데이터 경로
python -m risk_model data/ds1.csv data/ds2.csv data/ds3.csv

# 예측 모델 포함 (preds.csv 필요)
python -m risk_model data/ds1.csv data/ds2.csv data/ds3.csv data/preds.csv
```

**출력 예시** (`risk_output.csv`):
```csv
ENCODED_MCT,TA_YM,Sales_Risk,Customer_Risk,Market_Risk,RiskScore,p_model,p_final,Alert
ABC123,2024-01,0.15,0.12,0.55,0.28,0.32,0.30,ORANGE
```

### 2. 모델 학습 (Jupyter Notebook)

```bash
# Jupyter 시작
jupyter notebook

# 노트북 실행
# - train_baseline_fixed.ipynb: 기본 모델 학습
# - train_full_ensemble.ipynb: 전체 앙상블 모델
```

### 3. API 사용

#### A. 즉시 위험도 분석 (quickscore)

```bash
curl -X POST http://localhost:8000/api/v1/predict/quickscore \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev_key_12345" \
  -d '{
    "industry_code": "치킨",
    "region_code": "강남구",
    "sales_1m": 15000000,
    "sales_3m_avg": 18000000,
    "cust_1m": 450,
    "cust_3m_avg": 500,
    "delivery_share": 0.7
  }'
```

**응답 예시:**
```json
{
  "id": "pred_abc123",
  "p_final": 0.35,
  "alert": "ORANGE",
  "risk_components": {
    "Sales_Risk": 0.08,
    "Customer_Risk": 0.05,
    "Market_Risk": 0.56
  },
  "explanations": [
    "최근 1→3개월 대비 매출 모멘텀 둔화",
    "지역/업종 시장 위험도 상회"
  ],
  "recommendations": [
    "💰 매출 개선: 프로모션 이벤트나 신메뉴 출시를 고려하세요",
    "📱 온라인 강화: 배달앱 외 자체 채널을 개발하세요"
  ]
}
```

#### B. 챗봇 대화형 분석

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "강남구에서 치킨집 하는데 지난달 매출 1500만원이야"
  }'
```

#### C. 자연어 파싱 (NLP)

```bash
curl -X POST http://localhost:8000/api/v1/nlp/parse \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "강남구 치킨집인데 배달 위주고 한 달 매출 1500만원, 고객 450명"
  }'
```

**응답:**
```json
{
  "industry_code": "치킨",
  "region_code": "강남구",
  "delivery_share": 0.8,
  "sales_1m": 15000000.0,
  "cust_1m": 450.0
}
```

---

## 📚 API 문서

### 인증

모든 API 요청에는 헤더 필요:
```
X-API-Key: your-api-key
```

### 주요 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| `GET` | `/api/v1/health` | 헬스체크 |
| `POST` | `/api/v1/predict/quickscore` | 즉시 위험도 분석 |
| `POST` | `/api/v1/predict/model` | ML 모델 기반 예측 |
| `POST` | `/api/v1/nlp/parse` | 자연어 파싱 |
| `POST` | `/api/v1/chat` | 챗봇 대화 |
| `GET` | `/api/v1/predict/history/{store_id}` | 예측 이력 조회 |
| `GET` | `/api/v1/admin/stats` | API 사용 통계 |

**상세 문서**: http://localhost:8000/api/docs (Swagger UI)

---

## 🐳 배포

### Docker 프로덕션 배포

```bash
# 1. 환경 변수 설정
cat > .env <<EOF
ENVIRONMENT=production
SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET=$(openssl rand -hex 32)
DATABASE_URL=postgresql://user:password@postgres:5432/sme_warning
SENTRY_DSN=your-sentry-dsn
EOF

# 2. 빌드 및 실행
docker-compose build --no-cache
docker-compose up -d

# 3. DB 마이그레이션 (필요시)
docker-compose exec api alembic upgrade head

# 4. 헬스체크
curl http://localhost:8000/api/v1/health
```

### Nginx 리버스 프록시 (선택)

`nginx.conf` 파일이 포함되어 있으며, docker-compose에서 자동 설정됩니다.

### 모니터링

- **Prometheus**: http://localhost:9090 (설정 필요)
- **Grafana**: http://localhost:3000 (설정 필요)
- **Sentry**: 환경 변수에 DSN 설정

---

## 🛠️ 개발

### 프로젝트 구조

```
model_test/
├── api/                    # FastAPI 애플리케이션
│   ├── routes/            # API 라우터
│   ├── service/           # 비즈니스 로직
│   ├── database.py        # DB 모델
│   ├── cache.py           # Redis 캐시
│   └── main.py            # 앱 진입점
├── data/                   # 데이터 파일
├── figures/                # 시각화 결과
├── config.py               # 전역 설정
├── pipeline.py             # 위험도 분석 파이프라인
├── risk_components.py      # 위험도 계산 로직
├── ensemble.py             # 앙상블 모델
├── preprocessing.py        # 전처리
├── alerting.py             # 경보 로직
├── utils.py                # 유틸리티
├── run.py                  # CLI 실행
├── Dockerfile              # Docker 이미지
├── docker-compose.yml      # 컨테이너 오케스트레이션
└── requirements.txt        # Python 의존성
```

### 로컬 개발 서버 실행

```bash
# 개발 모드 (auto-reload)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 또는 Docker 개발 환경
docker-compose -f docker-compose.dev.yml up
```

### 테스트 (추가 예정)

```bash
pytest tests/ -v
```

### 코드 포맷팅

```bash
black api/ *.py
flake8 api/ *.py
```

---

## 📊 데이터 구조

### 입력 데이터

#### 1. `big_data_set1_f.csv` (매장 기본 정보)
```csv
ENCODED_MCT,HPSN_MCT_ZCD_NM,HPSN_MCT_BZN_CD_NM,MCT_ME_D,...
```

#### 2. `ds2_monthly_usage.csv` (월별 거래)
```csv
ENCODED_MCT,TA_YM,RC_M1_SAA,RC_M1_TO_UE_CT,DLV_SAA_RAT,...
```

#### 3. `ds3_monthly_customers.csv` (월별 고객)
```csv
ENCODED_MCT,TA_YM,M12_MAL_1020_RAT,MCT_UE_CLN_REU_RAT,...
```

### 출력 데이터

#### `risk_output.csv`
```csv
ENCODED_MCT,TA_YM,Sales_Risk,Customer_Risk,Market_Risk,RiskScore,p_model,p_final,Alert
ABC123,2024-01-01,0.15,0.12,0.55,0.28,0.32,0.30,ORANGE
```

---

## 🔧 환경 변수

주요 환경 변수 (`.env` 파일):

```bash
# 환경 설정
ENVIRONMENT=production        # development/staging/production
DEBUG=false

# 보안
SECRET_KEY=your-secret-key
JWT_SECRET=your-jwt-secret
API_KEY_HEADER=X-API-Key

# 데이터베이스
DATABASE_URL=postgresql://user:password@postgres:5432/sme_warning

# Redis
REDIS_URL=redis://redis:6379/0
CACHE_TTL=3600

# 인증
ENABLE_AUTH=true

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000

# 모니터링
SENTRY_DSN=https://your-sentry-dsn
ENABLE_METRICS=true
```
---

## 🆘 문제 해결

### Q: "CSV 인코딩 해석 실패" 에러
**A**: 데이터 파일이 `data/` 디렉토리에 있는지 확인하고, UTF-8/CP949 인코딩을 시도합니다.

### Q: Docker 컨테이너가 시작되지 않음
**A**: 포트 충돌 확인 (`docker-compose down` 후 재시작)

### Q: API 키 인증 실패
**A**: `.env` 파일의 `ENABLE_AUTH=false`로 설정하거나, 올바른 API 키 사용

### Q: 모델 예측 결과가 이상함
**A**: `risk_output_trained.csv`가 최신인지 확인하고, 필요 시 재학습

---