# ==============================================================================
# Enterprise AI Chatbot — Makefile
# ==============================================================================
# Development workflow commands for Docker Compose orchestration
# ==============================================================================

.PHONY: up down logs backend-test clean build

# Default target
all: help

# -----------------------------------------------------------------------------
# Docker Compose Commands
# -----------------------------------------------------------------------------

# Start all services in detached mode
up:
	docker compose up --build -d
	@echo ""
	@echo "Services started successfully!"
	@echo "  Frontend: http://localhost:3000"
	@echo "  Backend:  http://localhost:8000"
	@echo "  MinIO:    http://localhost:9001"
	@echo "  API Docs: http://localhost:8000/docs"
	@echo ""

# Stop all services
down:
	docker compose down
	@echo "All services stopped."

# Follow logs from all services
logs:
	docker compose logs -f

# Follow logs from a specific service
logs-backend:
	docker compose logs -f backend

logs-frontend:
	docker compose logs -f frontend

# Rebuild and restart a specific service
rebuild-backend:
	docker compose up --build -d backend

rebuild-frontend:
	docker compose up --build -d frontend

# -----------------------------------------------------------------------------
# Testing Commands
# -----------------------------------------------------------------------------

# Run pytest inside the backend container
backend-test:
	docker compose exec backend pytest tests/ -v

# Run pytest with coverage
backend-test-cov:
	docker compose exec backend pytest tests/ -v --cov=app --cov-report=html

# -----------------------------------------------------------------------------
# Utility Commands
# -----------------------------------------------------------------------------

# Remove all containers, volumes, and images (full reset)
clean:
	docker compose down -v --remove-orphans
	@echo "All containers and volumes removed."

# Remove only volumes (preserves images)
clean-volumes:
	docker compose down -v
	@echo "All volumes removed."

# Build images without starting containers
build:
	docker compose build

# Verify docker-compose configuration
check:
	docker compose config

# Restart a specific service
restart-backend:
	docker compose restart backend

restart-frontend:
	docker compose restart frontend

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
help:
	@echo "Enterprise AI Chatbot — Available Commands"
	@echo ""
	@echo "  make up              Start all services (detached, with rebuild)"
	@echo "  make down            Stop all services"
	@echo "  make logs            Follow logs from all services"
	@echo "  make logs-backend    Follow logs from backend only"
	@echo "  make logs-frontend   Follow logs from frontend only"
	@echo ""
	@echo "  make backend-test    Run pytest tests in backend container"
	@echo "  make backend-test-cov Run pytest with coverage report"
	@echo ""
	@echo "  make build           Build images without starting containers"
	@echo "  make check           Validate docker-compose configuration"
	@echo "  make clean           Remove all containers, volumes, and orphans"
	@echo "  make clean-volumes   Remove volumes only"
	@echo "  make restart-backend Restart backend service"
	@echo "  make restart-frontend Restart frontend service"
	@echo ""
	@echo "  make help            Show this help message"