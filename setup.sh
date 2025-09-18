#!/bin/bash

# TorchWeave LLM Setup Script
# This script sets up the complete TorchWeave system with frontend and backend

set -e  # Exit on any error

echo "=========================================="
echo " TorchWeave LLM Setup Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker is installed"
}

# Check for GPU support
check_gpu() {
    print_status "Checking for GPU support..."
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            print_success "NVIDIA GPU detected"
            return 0
        else
            print_warning "nvidia-smi found but GPU not accessible"
        fi
    else
        print_warning "No NVIDIA GPU detected, will use CPU"
    fi
    return 1
}

# Create directory structure
setup_directories() {
    print_status "Setting up directory structure..."
    
    # Create frontend directory if it doesn't exist
    mkdir -p frontend
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        print_status "Creating .env file..."
        cp .env.example .env 2>/dev/null || cat > .env << EOF
# Model Configuration
HF_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
MAX_NEW_TOKENS=128
TEMPERATURE=0.7
TOP_K=50
TOP_P=0.9
SEED=42

# Scheduler Configuration
SCHEDULE_TICK_MS=15
MAX_BATCH=16

# Storage Configuration
ARTIFACT_DIR=/artifacts
ARTIFACT_MODEL_DIR=/artifacts/model
EOF
        print_success "Created .env file with default configuration"
    else
        print_success ".env file already exists"
    fi
}

# Clean up previous containers and volumes
cleanup_previous() {
    print_status "Cleaning up previous containers..."
    
    # Stop and remove containers
    docker-compose down --volumes --remove-orphans 2>/dev/null || true
    docker compose down --volumes --remove-orphans 2>/dev/null || true
    
    # Remove dangling images
    docker image prune -f &>/dev/null || true
    
    print_success "Cleanup completed"
}

# Build and start services
start_services() {
    local use_gpu=$1
    
    print_status "Building and starting TorchWeave services..."
    
    if [ "$use_gpu" = true ]; then
        print_status "Starting with GPU support..."
        
        # First run optimizer to download model
        print_status "Running model optimizer (this may take a few minutes)..."
        docker-compose --profile setup up --build optimizer
        
        # Then start GPU services
        print_status "Starting GPU-enabled services..."
        docker-compose --profile gpu up --build -d server-gpu frontend-gpu
        
    else
        print_status "Starting with CPU support..."
        
        # First run optimizer to download model  
        print_status "Running model optimizer (this may take a few minutes)..."
        docker-compose --profile setup up --build optimizer
        
        # Then start CPU services
        print_status "Starting CPU services..."
        docker-compose up --build -d server frontend
    fi
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for backend
    print_status "Waiting for backend server..."
    for i in {1..60}; do
        if curl -s http://localhost:8000/health &>/dev/null; then
            print_success "Backend server is ready"
            break
        fi
        if [ $i -eq 60 ]; then
            print_error "Backend server failed to start within 5 minutes"
            print_status "Checking backend logs:"
            docker-compose logs server || docker compose logs server
            exit 1
        fi
        sleep 5
    done
    
    # Wait for frontend
    print_status "Waiting for frontend server..."
    for i in {1..30}; do
        if curl -s http://localhost:3000/health &>/dev/null; then
            print_success "Frontend server is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "Frontend server failed to start within 2.5 minutes"
            print_status "Checking frontend logs:"
            docker-compose logs frontend || docker compose logs frontend
            exit 1
        fi
        sleep 5
    done
}

# Display final information
show_completion_info() {
    echo ""
    echo "=========================================="
    print_success "TorchWeave LLM is now running!"
    echo "=========================================="
    echo ""
    echo "🌐 Frontend (Web Interface): http://localhost:3000"
    echo "🚀 Backend API: http://localhost:8000"
    echo "📊 API Documentation: http://localhost:8000/docs"
    echo ""
    echo "Available endpoints:"
    echo "  • Health Check: http://localhost:8000/health"
    echo "  • Model Info: http://localhost:8000/model"
    echo "  • Available Models: http://localhost:8000/models/available"
    echo "  • Loaded Models: http://localhost:8000/models/loaded"
    echo ""
    echo "Management commands:"
    echo "  • View logs: docker-compose logs -f server"
    echo "  • Stop services: docker-compose down"
    echo "  • Restart: docker-compose restart"
    echo ""
    print_status "Open your browser and navigate to http://localhost:3000 to start using TorchWeave!"
}

# Show usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --gpu          Enable GPU support (requires NVIDIA Docker)"
    echo "  --cpu          Force CPU-only mode (default if no GPU detected)"
    echo "  --clean        Clean up previous installations before starting"
    echo "  --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                # Auto-detect GPU/CPU and start"
    echo "  $0 --gpu          # Force GPU mode"
    echo "  $0 --cpu          # Force CPU mode"
    echo "  $0 --clean --gpu  # Clean install with GPU support"
}

# Main execution
main() {
    local use_gpu=false
    local force_cpu=false
    local clean_install=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --gpu)
                use_gpu=true
                shift
                ;;
            --cpu)
                force_cpu=true
                shift
                ;;
            --clean)
                clean_install=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Check prerequisites
    check_docker
    
    # Determine GPU usage
    if [ "$force_cpu" = true ]; then
        use_gpu=false
        print_status "Forcing CPU mode as requested"
    elif [ "$use_gpu" = true ]; then
        if ! check_gpu; then
            print_error "GPU mode requested but no GPU available"
            exit 1
        fi
    else
        # Auto-detect GPU
        if check_gpu; then
            use_gpu=true
        fi
    fi
    
    # Setup directories
    setup_directories
    
    # Clean up if requested
    if [ "$clean_install" = true ]; then
        cleanup_previous
    fi
    
    # Start services
    start_services $use_gpu
    
    # Wait for services
    wait_for_services
    
    # Show completion info
    show_completion_info
}

# Run main function with all arguments
main "$@"
