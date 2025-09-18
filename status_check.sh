#!/bin/bash
# TorchWeave Service Status Check Script
# Run this script to diagnose your setup

echo "=== TorchWeave Service Status Check ==="
echo "Timestamp: $(date)"
echo

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print status with colors
print_status() {
    local status=$1
    local message=$2
    case $status in
        "OK")
            echo -e "${GREEN}✓ OK${NC}: $message"
            ;;
        "WARN")
            echo -e "${YELLOW}⚠ WARNING${NC}: $message"
            ;;
        "ERROR")
            echo -e "${RED}✗ ERROR${NC}: $message"
            ;;
        "INFO")
            echo -e "${BLUE}ℹ INFO${NC}: $message"
            ;;
    esac
}

# Check if Docker is running
echo "1. Docker System Check"
echo "----------------------"
if command -v docker &> /dev/null; then
    if docker info &> /dev/null; then
        print_status "OK" "Docker is running"
        docker --version
    else
        print_status "ERROR" "Docker is installed but not running. Start Docker first."
        exit 1
    fi
else
    print_status "ERROR" "Docker is not installed"
    exit 1
fi

echo

# Check if docker-compose is available
echo "2. Docker Compose Check"
echo "----------------------"
if command -v docker-compose &> /dev/null; then
    print_status "OK" "docker-compose is available"
    docker-compose --version
elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
    print_status "OK" "docker compose (built-in) is available"
    docker compose version
else
    print_status "ERROR" "Neither docker-compose nor docker compose is available"
    exit 1
fi

echo

# Check if docker-compose.yml exists
echo "3. Configuration Files Check"
echo "---------------------------"
if [ -f "docker-compose.yml" ]; then
    print_status "OK" "docker-compose.yml found"
else
    print_status "ERROR" "docker-compose.yml not found in current directory"
    echo "Current directory: $(pwd)"
    echo "Files in directory:"
    ls -la
    exit 1
fi

# Check for other important files
if [ -f "nginx.conf" ]; then
    print_status "OK" "nginx.conf found"
else
    print_status "WARN" "nginx.conf not found"
fi

if [ -f "index.html" ]; then
    print_status "OK" "index.html found"
else
    print_status "WARN" "index.html not found"
fi

echo

# Check running containers
echo "4. Container Status Check"
echo "------------------------"
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
fi

print_status "INFO" "Checking container status..."

# List all containers for this project
if $COMPOSE_CMD ps &> /dev/null; then
    echo "Container Status:"
    $COMPOSE_CMD ps
    echo
    
    # Check individual services
    services=("server" "model-manager" "frontend")
    for service in "${services[@]}"; do
        container_id=$($COMPOSE_CMD ps -q "$service" 2>/dev/null)
        if [ -n "$container_id" ]; then
            status=$(docker inspect --format '{{.State.Status}}' "$container_id" 2>/dev/null)
            if [ "$status" = "running" ]; then
                print_status "OK" "$service container is running (ID: $container_id)"
            else
                print_status "ERROR" "$service container exists but status is: $status"
            fi
        else
            print_status "ERROR" "$service container not found"
        fi
    done
else
    print_status "WARN" "No docker-compose project found or not running"
fi

echo

# Check port accessibility
echo "5. Port Accessibility Check"
echo "--------------------------"
ports=(3000 8000 8001)
for port in "${ports[@]}"; do
    if nc -z localhost $port 2>/dev/null; then
        print_status "OK" "Port $port is accessible"
    else
        print_status "ERROR" "Port $port is not accessible"
    fi
done

echo

# Check HTTP endpoints
echo "6. HTTP Endpoint Check"
echo "---------------------"
endpoints=(
    "http://localhost:3000"
    "http://localhost:8000/health"
    "http://localhost:8001/health"
)

for endpoint in "${endpoints[@]}"; do
    if curl -s --max-time 5 "$endpoint" > /dev/null 2>&1; then
        print_status "OK" "$endpoint is responding"
    else
        print_status "ERROR" "$endpoint is not responding"
    fi
done

echo

# Check logs for errors (if containers are running)
echo "7. Recent Logs Check"
echo "-------------------"
if $COMPOSE_CMD ps -q > /dev/null 2>&1; then
    print_status "INFO" "Checking recent logs for errors..."
    
    # Check for common error patterns in logs
    error_patterns=("error" "Error" "ERROR" "exception" "Exception" "failed" "Failed" "FAILED")
    
    for service in server model-manager frontend; do
        echo "=== $service logs (last 20 lines) ==="
        if $COMPOSE_CMD logs --tail=20 "$service" 2>/dev/null; then
            echo
        else
            print_status "WARN" "Could not retrieve logs for $service"
        fi
    done
else
    print_status "WARN" "Cannot check logs - containers not running"
fi

echo

# Recommendations
echo "8. Troubleshooting Recommendations"
echo "================================="

# Check if any containers are not running
non_running_services=()
for service in server model-manager frontend; do
    container_id=$($COMPOSE_CMD ps -q "$service" 2>/dev/null)
    if [ -n "$container_id" ]; then
        status=$(docker inspect --format '{{.State.Status}}' "$container_id" 2>/dev/null)
        if [ "$status" != "running" ]; then
            non_running_services+=("$service")
        fi
    else
        non_running_services+=("$service")
    fi
done

if [ ${#non_running_services[@]} -gt 0 ]; then
    echo "Services not running: ${non_running_services[*]}"
    echo
    echo "To start services:"
    echo "  $COMPOSE_CMD up -d"
    echo
    echo "To rebuild and start:"
    echo "  $COMPOSE_CMD up -d --build"
    echo
    echo "To check individual service logs:"
    for service in "${non_running_services[@]}"; do
        echo "  $COMPOSE_CMD logs $service"
    done
    echo
fi

echo "Additional commands:"
echo "  View all logs:           $COMPOSE_CMD logs"
echo "  Follow logs:            $COMPOSE_CMD logs -f"
echo "  Restart services:       $COMPOSE_CMD restart"
echo "  Stop all services:      $COMPOSE_CMD down"
echo "  Rebuild everything:     $COMPOSE_CMD up -d --build --force-recreate"
echo
echo "Access URLs (if services are running):"
echo "  Frontend:               http://localhost:3000"
echo "  TorchWeave API:         http://localhost:8000"
echo "  Model Manager API:      http://localhost:8001"

echo
echo "=== Status Check Complete ==="
