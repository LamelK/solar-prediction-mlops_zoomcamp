#!/bin/bash

# Exit on any error
set -e

echo "📊 Starting Monitoring Container..."

# Start the monitoring service
echo "🔍 Starting Evidently AI drift monitoring..."
python -u monitoring/monitor_drift.py 