# LLM Tuner - Architecture

## Overview
Hardware-agnostic inference engine benchmarking and optimization framework with web UI.

## Tech Stack
- **Backend**: FastAPI + Uvicorn (Python)
- **Database**: SQLite (portable, no external deps)
- **Frontend**: Vanilla JS + Chart.js (no build step needed)
- **Deployment**: Docker-ready, runs on any GPU platform

## Hardware Abstraction Layer
```
HardwareDetector -> detects GPU vendor (ROCm/CUDA/Metal)
                   -> returns capabilities dict
                   
EngineFactory    -> builds engine config based on hardware
                   -> selects appropriate binaries/flags
```

## API Endpoints
- GET /api/hardware - Detect and return hardware profile
- POST /api/benchmarks - Start a benchmark run
- GET /api/benchmarks/:id - Get benchmark status/results
- GET /api/history - List past benchmarks
- DELETE /api/history/:id - Delete benchmark record
- GET /api/engines - List available engines for this hardware

## Benchmark Types
1. **Quick** - Single context length, single engine
2. **Full** - Multiple context lengths, all supported engines
3. **AI-Tune** - Model proposes its own optimal flags (8 rounds)
4. **Grid Search** - Systematic parameter space exploration
