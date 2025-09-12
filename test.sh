#!/bin/bash
# Test script for TorchWeave LLM services

echo "Testing TorchWeave LLM Services"
echo "================================"

# Test server health
echo "1. Testing server health..."
curl -s http://localhost:8000/health | jq .
echo

# Test model-manager health
echo "2. Testing model-manager health..."
curl -s http://localhost:8001/health | jq .
echo

# Test model info
echo "3. Testing model information..."
curl -s http://localhost:8000/model | jq .
echo

# Test model listing (initially empty)
echo "4. Testing model listing..."
curl -s http://localhost:8001/models | jq .
echo

# Test loading a model via structured endpoint
echo "5. Testing model loading (structured)..."
curl -s -X POST http://localhost:8001/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_config": {
      "model_id": "gpt2",
      "source": "huggingface",
      "display_name": "GPT-2 Base",
      "description": "Small GPT-2 for testing"
    }
  }' | jq .
echo

# Test loading a model via flat endpoint (backward compatibility)
echo "6. Testing model loading (flat/backward compatible)..."
timeout 30 curl -s -X POST http://localhost:8001/models/load_flat \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "source": "huggingface",
    "display_name": "TinyLlama Chat",
    "description": "TinyLlama model for testing"
  }' | jq .
if [ $? -eq 124 ]; then
    echo "Request timed out - model loading may take longer"
fi
echo

# List models after loading
echo "7. Listing models after loading..."
curl -s http://localhost:8001/models | jq .
echo

# Test text generation (streaming)
echo "8. Testing streaming text generation..."
echo "Sending request for streaming generation..."
curl -N -X POST http://localhost:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "The future of artificial intelligence is",
    "max_new_tokens": 32,
    "temperature": 0.7
  }' | head -20
echo

# Test text generation (non-streaming baseline)
echo "9. Testing non-streaming text generation..."
curl -s -X POST http://localhost:8000/v1/generate_nobatch \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "The future of artificial intelligence is",
    "max_new_tokens": 32,
    "temperature": 0.7
  }' | jq .
echo

# Test server status from model-manager
echo "10. Testing server status check..."
curl -s http://localhost:8001/server/status | jq .
echo

echo "Testing complete!"
echo "=================="
echo "If all tests passed, your TorchWeave LLM system is working correctly."
echo
echo "Available endpoints:"
echo "- Server: http://localhost:8000"
echo "- Model Manager: http://localhost:8001"
echo "- API Documentation: http://localhost:8000/docs (FastAPI auto-docs)"
echo "- Model Manager Docs: http://localhost:8001/docs"
