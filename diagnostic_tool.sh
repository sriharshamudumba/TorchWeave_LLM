#!/bin/bash
# TorchWeave Container Health Check Script

echo "🔍 TorchWeave Container Health Check"
echo "====================================="

# Check container status
echo "1. Container Status:"
docker-compose ps

echo ""
echo "2. Port Bindings:"
docker-compose ps --format "table {{.Name}}\t{{.Ports}}\t{{.Status}}"

echo ""
echo "3. Container Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

echo ""
echo "4. TorchWeave Server Logs (last 20 lines):"
echo "----------------------------------------"
docker logs --tail=20 torchweave_server

echo ""
echo "5. Model Manager Logs (last 20 lines):"
echo "--------------------------------------"
docker logs --tail=20 torchweave_model_manager

echo ""
echo "6. Network Connectivity Test:"
echo "----------------------------"

# Test internal container connectivity
echo "Testing internal connectivity..."
docker exec torchweave_nginx curl -s -o /dev/null -w "Server (8000): %{http_code}\n" http://server:8000/health || echo "Server (8000): Connection failed"
docker exec torchweave_nginx curl -s -o /dev/null -w "Model Manager (8001): %{http_code}\n" http://model-manager:8001/health || echo "Model Manager (8001): Connection failed"

echo ""
echo "Testing external connectivity..."
curl -s -o /dev/null -w "Server (localhost:8000): %{http_code}\n" http://localhost:8000/health || echo "Server (localhost:8000): Connection failed"
curl -s -o /dev/null -w "Model Manager (localhost:8001): %{http_code}\n" http://localhost:8001/health || echo "Model Manager (localhost:8001): Connection failed"
curl -s -o /dev/null -w "Frontend (localhost:3000): %{http_code}\n" http://localhost:3000/ || echo "Frontend (localhost:3000): Connection failed"

echo ""
echo "7. Environment Variables Check:"
echo "------------------------------"
echo "TorchWeave Server environment:"
docker exec torchweave_server printenv | grep -E "(PYTHONPATH|HF_|CUDA|TORCH)" || echo "No relevant env vars found"

echo ""
echo "Model Manager environment:"
docker exec torchweave_model_manager printenv | grep -E "(PYTHONPATH|HF_|CUDA|TORCH)" || echo "No relevant env vars found"

echo ""
echo "8. Python Process Check:"
echo "------------------------"
echo "TorchWeave Server processes:"
docker exec torchweave_server ps aux | grep python || echo "No Python processes found"

echo ""
echo "Model Manager processes:"
docker exec torchweave_model_manager ps aux | grep python || echo "No Python processes found"

echo ""
echo "9. File System Check:"
echo "--------------------"
echo "TorchWeave Server working directory:"
docker exec torchweave_server pwd
docker exec torchweave_server ls -la

echo ""
echo "Model Manager working directory:"
docker exec torchweave_model_manager pwd
docker exec torchweave_model_manager ls -la

echo ""
echo "10. Manual Service Test:"
echo "------------------------"
echo "Testing if we can manually start the services..."

echo "Attempting to import required modules in TorchWeave Server:"
docker exec torchweave_server python -c "
try:
    import torch
    print('✓ PyTorch imported successfully')
    import transformers
    print('✓ Transformers imported successfully')
    import fastapi
    print('✓ FastAPI imported successfully')
    from model_runtime import ModelRuntime
    print('✓ ModelRuntime imported successfully')
    from scheduler import ContinuousBatchingScheduler
    print('✓ Scheduler imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
except Exception as e:
    print(f'✗ Other error: {e}')
"

echo ""
echo "Testing Model Manager imports:"
docker exec torchweave_model_manager python -c "
try:
    import torch
    print('✓ PyTorch imported successfully')
    import transformers
    print('✓ Transformers imported successfully')
    import fastapi
    print('✓ FastAPI imported successfully')
    print('✓ All imports successful')
except ImportError as e:
    print(f'✗ Import error: {e}')
except Exception as e:
    print(f'✗ Other error: {e}')
"

echo ""
echo "====================================="
echo "Health check complete!"
echo ""
echo "🔧 Quick Fixes:"
echo "- If containers are exiting: Check the logs above for Python errors"
echo "- If ports are not bound: Check docker-compose.yml port configuration"
echo "- If imports fail: Check Dockerfile dependencies and Python environment"
echo "- If services are running but not responding: Check firewall/networking"
echo ""
echo "📋 Next Steps:"
echo "1. Fix any import errors shown above"
echo "2. Restart failed containers: docker-compose restart <service>"
echo "3. If issues persist: docker-compose down && docker-compose up -d"
echo "4. Run diagnostics again: python diagnostic_tool.py"
