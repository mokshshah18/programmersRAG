#!/bin/bash

# can run directly too don't need this

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if backend.py exists
if [ ! -f "backend.py" ]; then
    echo "backend.py not found. Please run setup.sh first."
    exit 1
fi

echo "Starting RAG server..."
echo "Server will be available at: http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo ""

# Start the backend server
python backend.py