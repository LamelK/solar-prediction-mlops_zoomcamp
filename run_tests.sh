#!/bin/bash
set -e

# Lint with flake8
flake8 .

# Run all tests with pytest and show coverage
pytest --cov=. 