# LLM Tuner - Inference Engine Benchmark Suite

Hardware-agnostic benchmarking and optimization framework for LLM inference engines with web UI.

## Features

- **Multi-engine support**: Compare llama.cpp, vLLM, SGLang side-by-side
- **AI-assisted tuning**: Model proposes its own optimal flags (8+ rounds)
- **Grid search**: Systematic parameter space exploration
- **Hardware detection**: Auto-detects ROCm/CUDA/Metal and configures accordingly
- **Persistent history**: SQLite database tracks all benchmarks with hardware state
- **Web dashboard**: Real-time charts, comparisons, and result visualization

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app (port 8090)
python3 app.py

# Or with uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8090 --reload
```

Open http://localhost:8090 in your browser.

## Benchmark Types

1. **Quick** - Single context length, fast results
2. **Full** - Multiple context lengths (512, 1024, 2048, 4096)
3. **AI-Assisted Tuning** - Model proposes optimal flags across multiple rounds
4. **Grid Search** - Systematic exploration of threads, batch sizes, cache types

## Hardware Support

| Platform | GPU Backend | Status |
|----------|-------------|--------|
| AMD ROCm | llama.cpp (HIP) | ✅ Production |
| NVIDIA CUDA | llama.cpp, vLLM, SGLang | ✅ Production |
| Apple Silicon | llama.cpp (Metal) | 🔄 Experimental |

## API Endpoints

- `GET /api/hardware` - Detect and return hardware profile
- `POST /api/benchmarks` - Start a benchmark run
- `GET /api/benchmarks/:id` - Get benchmark status/results
- `GET /api/history` - List past benchmarks
- `DELETE /api/history/:id` - Delete benchmark record

## Project Structure

```
llm-tuner/
├── app.py                    # FastAPI backend + web server
├── requirements.txt          # Python dependencies
├── static/                   # Frontend assets
│   ├── index.html           # Main UI
│   ├── styles.css           # Minimalist styling
│   └── app.js               # Chart.js visualizations
├── hardware/                # Hardware detection layer
│   └── detector.py          # Vendor-agnostic GPU profiling
├── database/                # Persistent storage
│   └── __init__.py          # SQLite benchmark manager
├── engines/                 # Inference engine backends
│   ├── base.py              # Abstract interface
│   └── llama_cpp.py         # Llama.cpp with ROCm/CUDA
└── tuners/                  # Optimization strategies
    ├── ai_tuner.py          # Model-as-optimizer
    └── grid_search.py       # Systematic exploration
```

## License

MIT
