# TorchWeave Enhanced Makefile
.PHONY: help build up down logs clean test benchmark

# Default environment
ENV_FILE ?= .env

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Run initial setup
	@echo "Setting up TorchWeave Enhanced..."
	python3 enhanced_setup.py

build: ## Build all Docker images
	docker compose build

up: ## Start services (CPU mode)
	docker compose up -d optimizer server-cpu

up-gpu: ## Start services (GPU mode)
	docker compose --profile gpu up -d optimizer server

up-full: ## Start all services including monitoring
	docker compose --profile gpu --profile monitoring --profile cache up -d

up-dev: ## Start in development mode with UI
	docker compose --profile ui up -d optimizer server-cpu ui

down: ## Stop all services
	docker compose down

logs: ## View logs from all services
	docker compose logs -f

logs-server: ## View server logs only
	docker compose logs -f server-cpu

logs-optimizer: ## View optimizer logs only
	docker compose logs -f optimizer

logs-ui: ## View UI logs only
	docker compose logs -f ui

clean: ## Clean up containers and volumes
	docker compose down -v --remove-orphans
	docker system prune -f

clean-all: ## Clean everything including images
	docker compose down -v --remove-orphans
	docker system prune -af

test: ## Run benchmarks
	python scripts/bench.py --concurrency 4 --iters 20 --sse http://localhost:8000/v1/generate

benchmark: ## Run comprehensive benchmarks
	@echo "Running continuous batching benchmark..."
	python scripts/bench.py --concurrency 8 --iters 24 --sse http://localhost:8000/v1/generate
	@echo "Running baseline benchmark..."
	python scripts/bench.py --concurrency 1 --iters 24 --nobatch http://localhost:8000/v1/generate_nobatch

health: ## Check service health
	@echo "Checking server health..."
	curl -f http://localhost:8000/health
	@echo "\nChecking model info..."
	curl -f http://localhost:8000/model

install-deps: ## Install local dependencies for development
	pip install -r server/requirements.txt

dev-server: ## Run server locally for development
	cd server && python -m uvicorn src.server:app --reload --host 0.0.0.0 --port 8000

monitor: ## Open monitoring dashboards
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3001 (admin/admin)"

status: ## Show service status
	docker compose ps

restart: ## Restart all services
	docker compose restart

restart-server: ## Restart server only
	docker compose restart server

# Environment-specific targets
prod: ## Production deployment
	ENV_FILE=.env.prod docker compose --profile gpu --profile monitoring up -d

staging: ## Staging deployment  
	ENV_FILE=.env.staging docker compose --profile gpu up -d

# Maintenance
update: ## Pull latest images and restart
	docker compose pull
	docker compose up -d

backup: ## Backup model artifacts
	docker run --rm -v torchweave_llm_artifacts:/data -v $(PWD)/backups:/backup alpine tar czf /backup/artifacts-$(shell date +%Y%m%d-%H%M%S).tar.gz -C /data .

restore: ## Restore model artifacts (use BACKUP_FILE=filename)
	docker run --rm -v torchweave_llm_artifacts:/data -v $(PWD)/backups:/backup alpine tar xzf /backup/$(BACKUP_FILE) -C /data
