#!/bin/bash

# Download required Ollama models

echo "Downloading Ollama models..."

echo "Checking if Ollama is running..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Ollama is not running. Please start it first:"
    echo "  ollama serve"
    exit 1
fi

echo "Pulling gemma3:latest..."
ollama pull gemma3:latest

echo "Pulling nomic-embed-text..."
ollama pull nomic-embed-text

echo "All models downloaded successfully"
echo ""
echo "Available models:"
ollama list