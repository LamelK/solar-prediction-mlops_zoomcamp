#!/bin/bash
set -e

# Lint with flake8
flake8 .

# Run all tests with pytest and show coverage, skipping integration tests
pytest --cov=. -m "not integration" 