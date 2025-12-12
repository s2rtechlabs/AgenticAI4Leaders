#!/bin/bash

# ============================================================
# A2A Exercise - Run All Agents
# ============================================================
# This script starts all three agents in separate processes.
# Use Ctrl+C to stop all agents.
# ============================================================

echo ""
echo "============================================================"
echo "  A2A Exercise - Starting All Agents"
echo "============================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Stopping all agents...${NC}"
    kill $(jobs -p) 2>/dev/null
    echo -e "${GREEN}All agents stopped.${NC}"
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT SIGTERM

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Install dependencies if needed
echo "Checking dependencies..."
pip install -q fastapi uvicorn httpx pydantic 2>/dev/null

echo ""
echo "Starting agents..."
echo ""

# Start Calculator Agent
echo -e "${GREEN}[1/3]${NC} Starting Calculator Agent on port 10001..."
cd "$SCRIPT_DIR/agents/calculator" && python3 -m __main__ &
sleep 1

# Start Greeter Agent
echo -e "${GREEN}[2/3]${NC} Starting Greeter Agent on port 10002..."
cd "$SCRIPT_DIR/agents/greeter" && python3 -m __main__ &
sleep 1

# Start Weather Agent
echo -e "${GREEN}[3/3]${NC} Starting Weather Agent on port 10003..."
cd "$SCRIPT_DIR/agents/weather" && python3 -m __main__ &
sleep 1

echo ""
echo "============================================================"
echo -e "${GREEN}All agents are running!${NC}"
echo "============================================================"
echo ""
echo "Agent URLs:"
echo "  • Calculator: http://localhost:10001"
echo "  • Greeter:    http://localhost:10002"
echo "  • Weather:    http://localhost:10003"
echo ""
echo "To test agent discovery:"
echo "  curl http://localhost:10001/.well-known/agent.json | python -m json.tool"
echo ""
echo "To run the client:"
echo "  cd client && python simple_client.py"
echo ""
echo "Press Ctrl+C to stop all agents..."
echo ""

# Wait for all background processes
wait


