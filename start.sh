#!/bin/bash
# LLM Tuner - Startup Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Install dependencies if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -q -r requirements.txt
fi

PORT=${PORT:-8090}
echo "Starting LLM Tuner on port $PORT"
echo "Open http://localhost:$PORT in your browser"

# Start the server
python3 app.py --port "$PORT"
