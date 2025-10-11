# TorchWeave LLM – High-Performance Inference Server

TorchWeave LLM is a modular, high-performance inference platform designed for efficient large language model (LLM) serving. It supports continuous batching, dynamic model management, and real-time token streaming through a fully containerized microservices architecture.

---

## Features

### Core Performance
- **Continuous Batching** – Dynamically batches incoming requests for optimal GPU utilization  
- **KV-Cache Optimization** – Manages per-request cache to reduce redundant computation  
- **Dynamic Model Loading** – Loads and unloads models at runtime via Hugging Face integration  
- **Real-Time Streaming** – Supports Server-Sent Events (SSE) for token-level streaming responses  

### Infrastructure
- **Microservices Architecture** – Modular components managed through Docker Compose  
- **Redis Integration** – Provides distributed caching and session tracking  
- **Model Manager** – Handles model lifecycle, including search, load, and unload operations  
- **Web Interface** – Offers an interactive UI for model testing and management  

### Developer Experience
- **REST API** – Built with FastAPI and documented via OpenAPI/Swagger  
- **Health Monitoring** – Includes endpoint-based service and system health checks  
- **CI/CD Pipeline** – Uses GitHub Actions for automated testing and deployment  

---

## System Architecture

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

---

## Quick Start

### Prerequisites
- Docker and Docker Compose  
- Python 3.12 or newer (for local development)  
- CUDA-compatible GPU (recommended for optimal performance)  

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

### Service Access
- Web Interface: http://localhost:3000  
- Main API: http://localhost:8000  
- Model Manager: http://localhost:8001  
- API Documentation: http://localhost:8000/docs

---

## API Examples

### Load a Model
```bash
curl -X POST http://localhost:8001/models/load   -H "Content-Type: application/json"   -d '{"model_id": "microsoft/DialoGPT-medium"}'
```

### Generate Text
```bash
curl -X POST http://localhost:8000/generate   -H "Content-Type: application/json"   -d '{
    "prompt": "Hello, how are you?",
    "max_length": 50,
    "temperature": 0.7
  }'
```

### Stream Responses (SSE)
```bash
curl -N http://localhost:8000/stream   -H "Accept: text/event-stream"   -d '{"prompt": "Tell me a story", "max_length": 100}'
```

### Search Models on Hugging Face
```bash
curl http://localhost:8001/models/search/llama?limit=5
```

---

## Service Overview

### Main Server (`server/`)
- FastAPI service handling inference requests  
- Implements continuous batching for throughput optimization  
- Supports SSE-based real-time token streaming  
- Manages per-request KV-cache for efficient memory usage  

### Model Manager (`model-manager/`)
- Dynamically loads models from the Hugging Face Hub  
- Handles model lifecycle operations and health monitoring  
- Provides model discovery and search APIs  

### Web Interface (`ui/`)
- Offers an interactive chat-style interface for testing  
- Includes dashboards for model and system status  
- Visualizes token streaming in real time  

### Optimizer Service
- Performs runtime model optimization  
- Profiles model performance and resource usage  
- Reports metrics for fine-tuning deployment configurations  

---

## Development Setup

### Local Development
```bash
python -m venv llm_env
source llm_env/bin/activate
pip install -r requirements.txt

python -m src.server
python -m src.model_manager
```

### Running Tests
```bash
pytest tests/
```

### Code Quality Checks
```bash
black src/
isort src/
mypy src/
flake8 src/
```

---

## Performance Features

### Continuous Batching
- Groups incoming requests dynamically based on GPU availability  
- Manages KV-cache per request to reduce redundancy  
- Adapts batch size in real time based on model load  

### Memory Optimization
- Implements efficient attention operations  
- Prunes and reuses cache entries for memory savings  
- Uses memory-mapped loading for large models  

### Streaming
- Streams tokens as they are generated using SSE  
- Minimizes time-to-first-token (TTFT)  
- Allows configurable streaming parameters  

---

## Monitoring and Health

### Health Endpoints
- Main Server: `GET /health`  
- Model Manager: `GET /health`  
- System Status: `GET /status`  

### Metrics and Logging
- Tracks request latency and throughput  
- Monitors token generation rates  
- Logs resource utilization and model statistics  

---

## Contributing

1. Fork the repository  
2. Create a feature branch:  
   ```bash
   git checkout -b feature/my-feature
   ```  
3. Commit your changes:  
   ```bash
   git commit -m "Add my feature"
   ```  
4. Push the branch:  
   ```bash
   git push origin feature/my-feature
   ```  
5. Open a Pull Request  

---

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

TorchWeave LLM builds on the following open-source technologies:
- Hugging Face Transformers  
- FastAPI  
- PyTorch  
- Redis  

---

## Support

- **Documentation:** Available via the `/docs` endpoint  
- **Issues:** Use GitHub Issues for bug reports and feature requests  
- **Discussions:** Participate in ongoing conversations in the project’s Discussions tab  

---

**TorchWeave LLM — built for reliable, high-performance language model inference.**
