#!/bin/bash

# RAG System Setup Script
set -e

# Check if Python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $(pwd)"

# Check if virtual environment already exists
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip first
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing Python dependencies..."
pip install fastapi uvicorn faiss-cpu numpy ollama

echo "All dependencies installed successfully"

# Create default config file if it doesn't exist
if [ ! -f "config" ]; then
    echo "Creating default config file..."
    echo "# Add your file paths here (one per line)" > config
    echo "# Example:" >> config
    echo "# /path/to/your/file1.cpp" >> config
    echo "# /path/to/your/file2.py" >> config
    echo "Default config file created"
else
    echo "Config file already exists"
fi

# Create frontend directory if it doesn't exist
if [ ! -d "frontend" ]; then
    echo "Creating frontend directory..."
    mkdir -p frontend
    echo "Frontend directory created"
fi

echo ""
echo "======================================="
echo "       Setup Completed Successfully!"
echo "======================================="
echo ""
echo "To start the RAG system:"
echo "  source .venv/bin/activate"
echo "  python backend.py"
echo ""
echo "Next steps:"
echo "  1. Edit the 'config' file and add your file paths"
echo "  2. Or run './find_files.sh' to auto-discover files"
echo "  3. Start the server: python backend.py"
echo "  4. Open http://localhost:8000 in your browser"
echo "  5. Click 'Reindex' to build the search index"
echo ""