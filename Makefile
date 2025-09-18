# TorchWeave Makefile

# Default target
.DEFAULT_GOAL := up

# Bring up all services
up:
	docker compose up -d
	@echo ""
	@echo "=============================================="
	@echo " TorchWeave Frontend is available at: http://localhost:3000"
	@echo " Backend Server (TorchWeave):      http://localhost:8000"
	@echo "Model Manager API:                 http://localhost:8001"
	@echo "=============================================="
	@echo ""

# Stop and remove containers, networks, volumes
down:
	docker compose down -v --remove-orphans

# Clean everything (containers, volumes, networks)
clean:
	docker compose down -v --remove-orphans || true
	docker ps -aq | xargs -r docker rm -f
	docker network prune -f
	docker volume prune -f

# Build all services without cache
build:
	docker compose build --no-cache

# Tail logs for all services
logs:
	docker compose logs -f --tail=100

# Health check all services
health:
	@echo ">>> Checking TorchWeave Server..."
	-@curl -s http://localhost:8000/health | jq . || echo " torchweave_server not healthy"
	@echo ">>> Checking Model Manager..."
	-@curl -s http://localhost:8001/health | jq . || echo " torchweave_model_manager not healthy"
	@echo ">>> Checking Frontend..."
	-@curl -s -I http://localhost:3000 || echo " torchweave_nginx not serving"

# Run tests if you have test.sh
test:
	./test.sh
