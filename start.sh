#!/bin/bash
# TorchWeave Startup Script
# Usage: ./start.sh [cpu|gpu] [--ui] [--monitoring]

set -e

# Default configuration
MODE="cpu"
ENABLE_UI=false
ENABLE_MONITORING=false
PROFILES=""

# Parse arguments
for arg in "$@"; do
    case $arg in
        cpu)
            MODE="cpu"
            ;;
        gpu)
            MODE="gpu"
            PROFILES="--profile gpu"
            ;;
        --ui)
            ENABLE_UI=true
            PROFILES="$PROFILES --profile ui"
            ;;
        --monitoring)
            ENABLE_MONITORING=true
            PROFILES="$PROFILES --profile monitoring"
            ;;
        --help)
            echo "TorchWeave LLM Startup Script"
            echo ""
            echo "Usage: $0 [cpu|gpu] [--ui] [--monitoring]"
            echo ""
            echo "Options:"
            echo "  cpu            Run with CPU inference (default)"
            echo "  gpu            Run with GPU inference"
            echo "  --ui           Enable web UI (port 3000)"
            echo "  --monitoring   Enable Prometheus/Grafana monitoring"
            echo ""
            echo "Examples:"
            echo "  $0 cpu --ui              # CPU inference with web UI"
            echo "  $0 gpu --ui --monitoring # Full GPU setup with monitoring"
            echo "  $0                       # Basic CPU setup"
            exit 0
            ;;
    esac
done

# Load environment variables if .env exists
if [[ -f .env ]]; then
    echo "Loading environment from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Check for required directories
echo "Checking directory structure..."
required_dirs=("server" "optimizer" "model-manager" "ui")
for dir in "${required_dirs[@]}"; do
    if [[ ! -d "$dir" ]]; then
        echo "Error: Required directory '$dir' not found"
        echo "Please ensure you're running this script from the TorchWeave root directory"
        exit 1
    fi
done

# Display configuration
echo "===========================================" 
echo "         TorchWeave LLM Starting"
echo "==========================================="
echo "Mode: $MODE"
echo "Model: ${HF_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
echo "UI Enabled: $ENABLE_UI"
echo "Monitoring Enabled: $ENABLE_MONITORING"
echo ""

# Build and start services
echo "Building and starting services..."

if [[ $MODE == "gpu" ]]; then
    echo "Starting with GPU support..."
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        echo "Warning: nvidia-smi not found. Make sure NVIDIA drivers are installed."
    fi
    docker-compose $PROFILES up --build server model-manager
else
    echo "Starting with CPU-only inference..."
    docker-compose $PROFILES up --build server-cpu model-manager
fi

# Provide access information
echo ""
echo "==========================================="
echo "         TorchWeave LLM Started"
echo "==========================================="
echo ""
echo "Services:"
echo "  Inference API:    http://localhost:8000"
echo "  Model Manager:    http://localhost:8001"

if [[ $ENABLE_UI == true ]]; then
    echo "  Web UI:           http://localhost:3000"
fi

if [[ $ENABLE_MONITORING == true ]]; then
    echo "  Prometheus:       http://localhost:9090"
    echo "  Grafana:          http://localhost:3001 (admin/admin)"
fi

echo ""
echo "API Endpoints:"
echo "  Health Check:     curl http://localhost:8000/health"
echo "  List Models:      curl http://localhost:8001/models"
echo "  Generate Text:    curl -X POST http://localhost:8000/v1/generate"
echo ""

if [[ $ENABLE_UI == false ]]; then
    echo "To use the enhanced web interface:"
    echo "  1. Save the HTML interface to a file"
    echo "  2. Open it in your browser"
    echo "  3. Or restart with --ui flag for integrated experience"
    echo ""
fi

echo "To stop services: docker-compose down"
echo "To view logs: docker-compose logs -f"
