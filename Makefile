# Makefile for code quality

.PHONY: lint format check deploy-prefect api-shell monitoring-shell api-debug monitoring-debug api-up monitoring-up test install-hooks

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

api-shell:
	docker exec -it solar-api-service /bin/bash || docker exec -it solar-api-service /bin/sh

monitoring-shell:
	docker exec -it solar-monitoring /bin/bash || docker exec -it solar-monitoring /bin/sh

api-debug:
	docker run --rm -it --entrypoint /bin/bash solar-prediction-mlops_zoomcamp-api-service || docker run --rm -it --entrypoint /bin/sh solar-prediction-mlops_zoomcamp-api-service

monitoring-debug:
	docker run --rm -it --entrypoint /bin/bash solar-prediction-mlops_zoomcamp-monitoring || docker run --rm -it --entrypoint /bin/sh solar-prediction-mlops_zoomcamp-monitoring

api-up:
	docker compose up api-service

monitoring-up:
	docker compose up monitoring

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