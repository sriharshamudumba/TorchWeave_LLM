# TorchWeave LLM: High-Performance Inference Server


[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![Docker](https://img.shields.io/badge/docker-compose-blue.svg)](https://docs.docker.com/compose/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A high-performance distributed LLM inference server featuring continuous batching, dynamic model management, KV-cache optimization, and real-time streaming capabilities.

## 🚀 Features

### Core Performance
- **Continuous Batching**: Efficient request processing with dynamic batching
- **KV-Cache Optimization**: Per-request cache management for memory efficiency
- **Dynamic Model Loading**: Runtime model management with HuggingFace Hub integration
- **Real-time Streaming**: Server-Sent Events (SSE) for token-by-token delivery

### Infrastructure
- **Microservices Architecture**: Containerized services with Docker Compose
- **Redis Integration**: Distributed caching and session management  
- **Model Manager**: Dedicated service for model lifecycle management
- **Web Interface**: Interactive UI for model interaction and management

### Developer Experience
- **REST API**: OpenAPI/Swagger documentation
- **Health Monitoring**: Comprehensive health checks and status endpoints
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web UI        │    │  Main Server    │    │ Model Manager   │
│   (Port 3000)   │◄──►│  (Port 8000)    │◄──►│  (Port 8001)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Redis Cache    │
                       │  (Port 6379)    │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.12+ (for local development)
- CUDA-compatible GPU (recommended)

### Using Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/sriharshamudumba/TorchWeave_LLM.git
   cd TorchWeave_LLM
   ```

2. **Start all services**
   ```bash
   docker-compose up -d
   ```

3. **Verify deployment**
   ```bash
   docker-compose ps
   ```

### Access Points
- **Web Interface**: http://localhost:3000
- **Main API**: http://localhost:8000
- **Model Manager**: http://localhost:8001
- **API Documentation**: http://localhost:8000/docs

## 🔧 API Usage

### Load a Model
```bash
curl -X POST http://localhost:8001/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_id": "microsoft/DialoGPT-medium"}'
```

### Generate Text
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, how are you?",
    "max_length": 50,
    "temperature": 0.7
  }'
```

### Stream Responses (SSE)
```bash
curl -N http://localhost:8000/stream \
  -H "Accept: text/event-stream" \
  -d '{"prompt": "Tell me a story", "max_length": 100}'
```

### Search HuggingFace Models
```bash
curl http://localhost:8001/models/search/llama?limit=5
```

## 📖 Service Details

### Main Server (`server/`)
- **FastAPI application** handling inference requests
- **Continuous batching** for optimal throughput
- **SSE streaming** for real-time token delivery
- **KV-cache management** for memory efficiency

### Model Manager (`model-manager/`)
- **Dynamic model loading** from HuggingFace Hub
- **Model lifecycle management** (load, unload, monitor)
- **Health monitoring** and progress tracking
- **Model search** and discovery

### Web Interface (`ui/`)
- **Interactive chat interface** for model testing
- **Model management dashboard** 
- **Real-time streaming** visualization
- **System monitoring** and health checks

### Optimizer Service
- **Model optimization** techniques
- **Performance profiling** and metrics
- **Resource utilization** monitoring

## 🛠️ Development

### Local Setup
```bash
# Create virtual environment
python -m venv llm_env
source llm_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run individual services
python -m src.server
python -m src.model_manager
```

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black src/
isort src/

# Type checking
mypy src/

# Linting
flake8 src/
```

## 📊 Performance Features

### Continuous Batching
- Dynamic request batching for maximum GPU utilization
- Per-request KV-cache management
- Adaptive batch sizing based on model capacity

### Memory Optimization
- Efficient attention mechanism implementation
- KV-cache pruning and management
- Memory-mapped model loading

### Streaming Capabilities
- Real-time token delivery via SSE
- Time-to-first-token optimization
- Configurable streaming parameters

## 🔍 Monitoring

### Health Endpoints
- **Server Health**: `GET /health`
- **Model Manager Health**: `GET http://localhost:8001/health`
- **System Status**: `GET http://localhost:8001/status`

### Metrics & Logging
- Request/response latency tracking
- Token generation speed monitoring
- Resource utilization metrics
- Comprehensive logging with structured format

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **HuggingFace Transformers** for model implementations
- **FastAPI** for the web framework
- **PyTorch** for deep learning capabilities
- **Redis** for caching and session management

## 📞 Support

- **Documentation**: Check the `/docs` endpoints for API documentation
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join project discussions for feature requests

---

**Built with ❤️ for high-performance LLM inference**
