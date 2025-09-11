#!/bin/bash

# Camera Alignment Web App Launcher

echo "Starting Camera Alignment Web App..."
echo "=================================="

# Navigate to the backend directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/backend"

# Start the server
echo "Starting server on http://localhost:8000"
echo "Open your browser and navigate to http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================="

python main.py