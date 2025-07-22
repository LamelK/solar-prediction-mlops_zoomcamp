# Makefile for code quality

.PHONY: lint format check deploy-prefect

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