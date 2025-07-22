#!/bin/bash
# This script loads environment variables from .env and starts the Prefect worker named 'ml-pool'.
# Place your .env file in the same directory as this script or update the path below.

# Exit immediately if a command exits with a non-zero status
set -e

# Load all variables from .env into the environment
set -a
source "$(dirname "$0")/.env"
set +a

# Start the Prefect worker with the specified name
prefect worker start --pool ml-pool 