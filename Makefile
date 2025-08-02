# Makefile for code quality

.PHONY: lint format check deploy-prefect test install-hooks

lint:
	flake8 .

format:
	black .
	ruff check . --fix

check:
	black --check .
	ruff check . 

deploy-prefect:
	python $(CURDIR)/prefect_deployment.py

test:
	flake8 .
	pytest --cov=.

test-unit:
	flake8 .
	pytest --cov=. -m "not integration"

install-hooks:
	chmod +x .git/hooks/pre-push

# Python virtual environment automation
.PHONY: venv venv-install venv-activate

# Create a new virtual environment in ./venv
venv:
	python3 -m venv venv

# Install requirements into the venv
venv-install: venv
	. ./venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

# Print activation command for the venv
venv-activate:
	@echo "Run: source ./venv/bin/activate"

# Docker management commands
.PHONY: docker-status docker-stop docker-start docker-restart docker-logs docker-clean docker-clean-all docker-debug docker-inspect docker-exec

# Show status of all containers
docker-status:
	docker ps -a

# Stop all project containers
docker-stop:
	docker stop solar-api-service solar-monitoring solar-grafana solar-prometheus

# Stop individual containers
docker-stop-api:
	docker stop solar-api-service

docker-stop-monitoring:
	docker stop solar-monitoring

docker-stop-grafana:
	docker stop solar-grafana

docker-stop-prometheus:
	docker stop solar-prometheus

# Start all project containers
docker-start:
	docker start solar-api-service solar-monitoring solar-grafana solar-prometheus

# Start individual containers
docker-start-api:
	docker start solar-api-service

docker-start-monitoring:
	docker start solar-monitoring

docker-start-grafana:
	docker start solar-grafana

docker-start-prometheus:
	docker start solar-prometheus

# Restart all project containers
docker-restart:
	docker restart solar-api-service solar-monitoring solar-grafana solar-prometheus

# Show logs for all containers
docker-logs:
	docker logs solar-api-service
	docker logs solar-monitoring
	docker logs solar-grafana
	docker logs solar-prometheus

# Show logs for specific container
docker-logs-api:
	docker logs solar-api-service

docker-logs-monitoring:
	docker logs solar-monitoring

docker-logs-grafana:
	docker logs solar-grafana

docker-logs-prometheus:
	docker logs solar-prometheus

# Debug commands - inspect containers
docker-inspect:
	docker inspect solar-api-service
	docker inspect solar-monitoring
	docker inspect solar-grafana
	docker inspect solar-prometheus

# Debug commands - exec into running containers
docker-shell-api:
	docker exec -it solar-api-service /bin/bash || docker exec -it solar-api-service /bin/sh

docker-shell-monitoring:
	docker exec -it solar-monitoring /bin/bash || docker exec -it solar-monitoring /bin/sh

# Debug commands - show container stats
docker-stats:
	docker stats solar-api-service solar-monitoring solar-grafana solar-prometheus

# Debug commands - show container resource usage
docker-top:
	docker top solar-api-service
	docker top solar-monitoring
	docker top solar-grafana
	docker top solar-prometheus

# Debug commands - run new temporary containers for debugging
docker-debug-api:
	docker run -it --rm --entrypoint /bin/bash solar-prediction-mlops_zoomcamp-api-service || docker run -it --rm --entrypoint /bin/sh solar-prediction-mlops_zoomcamp-api-service

docker-debug-monitoring:
	docker run -it --rm --entrypoint /bin/bash solar-prediction-mlops_zoomcamp-monitoring || docker run -it --rm --entrypoint /bin/sh solar-prediction-mlops_zoomcamp-monitoring

# Clean up containers and images for this project only (only images currently running)
docker-clean:
	docker stop solar-api-service solar-monitoring solar-grafana solar-prometheus
	docker rm solar-api-service solar-monitoring solar-grafana solar-prometheus
	docker rmi solar-prediction-mlops_zoomcamp-api-service solar-prediction-mlops_zoomcamp-monitoring grafana/grafana:latest prom/prometheus:latest

# Clean up everything including project volumes
docker-clean-all:
	docker stop solar-api-service solar-monitoring solar-grafana solar-prometheus
	docker rm solar-api-service solar-monitoring solar-grafana solar-prometheus
	docker rmi solar-prediction-mlops_zoomcamp-api-service solar-prediction-mlops_zoomcamp-monitoring grafana/grafana:latest prom/prometheus:latest
	docker compose down -v

# Docker Compose commands
.PHONY: compose-up compose-down compose-restart compose-logs compose-build compose-pull

# Start all services
compose-up:
	docker compose up -d

# Stop all services
compose-down:
	docker compose down

# Restart all services
compose-restart:
	docker compose restart

# Show logs for all services
compose-logs:
	docker compose logs

# Show logs for specific service
compose-logs-api:
	docker compose logs api-service

compose-logs-monitoring:
	docker compose logs monitoring

compose-logs-grafana:
	docker compose logs grafana

compose-logs-prometheus:
	docker compose logs prometheus

# Build all services
compose-build:
	docker compose build

# Pull latest images
compose-pull:
	docker compose pull