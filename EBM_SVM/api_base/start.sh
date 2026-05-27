#!/bin/bash

# Development server startup script for Linux/Docker

# Set environment variables
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Change to api_base directory
cd "$(dirname "$0")"

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
fi

# Create necessary directories
mkdir -p logs utils/upload_temp utils/download models

# Install dependencies if needed
if command -v python3 &> /dev/null; then
    echo "Installing dependencies..."
    python3 -m pip install -r requirements.txt > /dev/null 2>&1
fi

# Start the server
echo "Starting EBM-SVM API server..."
python3 run_api.py "$@"
