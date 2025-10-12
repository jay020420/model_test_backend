# Makefile (편의 명령어)
.PHONY: help build up down logs test clean migrate

help:
	@echo "Available commands:"
	@echo "  make build    - Build Docker images"
	@echo "  make up       - Start services"
	@echo "  make down     - Stop services"
	@echo "  make logs     - View logs"
	@echo "  make test     - Run tests"
	@echo "  make migrate  - Run database migrations"
	@echo "  make clean    - Clean up containers and volumes"

build:
	docker-compose build

up:
	docker-compose up -d
	@echo "Services started. API available at http://localhost:8000"

down:
	docker-compose down

logs:
	docker-compose logs -f api

test:
	docker-compose exec api pytest tests/ -v

migrate:
	docker-compose exec api alembic upgrade head

clean:
	docker-compose down -v
	docker system prune -f

# 개발 환경
dev:
	docker-compose -f docker-compose.dev.yml up

# 프로덕션 배포
deploy:
	@echo "Deploying to production..."
	git pull origin main
	docker-compose build --no-cache
	docker-compose up -d
	docker-compose exec api alembic upgrade head
	@echo "Deployment complete!"

# 데이터베이스 백업
backup:
	docker-compose exec postgres pg_dump -U sme_user sme_warning > backup_$(shell date +%Y%m%d_%H%M%S).sql