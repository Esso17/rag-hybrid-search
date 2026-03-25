#!/bin/bash
# Quick setup script for RAG Hybrid Search

set -e

echo "🚀 RAG Hybrid Search - Setup"
echo "=============================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Docker
echo "📋 Checking prerequisites..."
echo ""

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found${NC}"
    echo "   Please install Docker: https://docker.com"
    exit 1
fi
echo -e "${GREEN}✅ Docker installed${NC}"

# Check Ollama
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Ollama not running${NC}"
    echo "   Start Ollama in another terminal:"
    echo "   $ ollama serve"
    echo ""
    echo "   Then pull models:"
    echo "   $ ollama pull mistral"
    echo "   $ ollama pull nomic-embed-text"
    echo ""
    read -p "Press Enter when Ollama is ready..."
fi
echo -e "${GREEN}✅ Ollama running${NC}"

# Check Python environment
echo ""
echo "🐍 Setting up Python environment..."
echo ""

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${GREEN}✅ Virtual environment exists${NC}"
fi

# Activate and install
echo "Installing dependencies..."
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Start Qdrant
echo ""
echo "🔧 Starting services..."
echo ""

echo "Starting Qdrant vector database..."
docker-compose up -d > /dev/null 2>&1
sleep 3

if curl -s http://localhost:6333/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Qdrant running${NC}"
else
    echo -e "${RED}❌ Qdrant failed to start${NC}"
    exit 1
fi

echo ""
echo "=============================="
echo -e "${GREEN}✅ Setup complete!${NC}"
echo "=============================="
echo ""
echo "Next steps:"
echo "1. Start the API server:"
echo "   $ source .venv/bin/activate"
echo "   $ python -m uvicorn app.main:app --reload"
echo ""
echo "2. Test the system:"
echo "   $ python scripts/test_ingestion.py"
echo ""
echo "3. API docs:"
echo "   http://localhost:8000/docs"
echo ""
