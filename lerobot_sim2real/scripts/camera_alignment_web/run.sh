#!/bin/bash

# Camera Alignment Web App Launcher

echo "Starting Camera Alignment Web App..."
echo "=================================="

# Navigate to the backend directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/backend"

# Check if env_config.json exists
ENV_CONFIG_PATH="$SCRIPT_DIR/../../../env_config.json"
if [ -f "$ENV_CONFIG_PATH" ]; then
    echo "Found env_config.json at $ENV_CONFIG_PATH"
    echo "Camera settings will be loaded from this file"
fi

# Start the server
echo "Starting server on http://localhost:8000"
echo "Open your browser and navigate to http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================="

python main.py