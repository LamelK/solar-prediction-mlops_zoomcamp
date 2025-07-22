# Makefile for code quality

.PHONY: lint format check

lint:
	flake8 .

format:
	black .
	ruff check . --fix

check:
	black --check .
	ruff check . 