# Enhanced Verilog Generation System with VerilogEval Integration

A comprehensive AI-powered hardware design pipeline with industry-standard VerilogEval quality assessment.

## 🎯 VerilogEval Features

- **Comprehensive Quality Assessment**: 100-point scoring system covering syntax, functionality, and design quality
- **Benchmark Comparison**: Compare designs against HDLBits reference problems
- **Pattern Recognition**: Identify and validate design patterns for different circuit types
- **Real-time Metrics**: Live quality metrics displayed throughout the design process
- **Automated Testing**: Functional correctness validation through simulation
- **Best Practices Compliance**: Ensure code follows established HDL design standards

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 16+ (for frontend)
- iverilog (for simulation)
- GROQ API key

### Installation

1. **Clone and setup Python backend:**
```bash
python setup_verilogeval_integration.py
cp .env.example .env
# Edit .env with your API keys
pip install -r requirements.txt
```

2. **Start the enhanced system:**
```bash
python run_enhanced_system.py
```

### Using Docker

```bash
docker-compose up --build
```

## 📊 VerilogEval Scoring System

| Component | Points | Description |
|-----------|---------|-------------|
| **Syntax Compliance** | 30 | Module structure, sensitivity lists, assignments |
| **Functional Correctness** | 40 | Compilation, simulation, waveform generation |
| **Design Quality** | 30 | Modularity, naming, patterns, documentation |
| **Total** | 100 | Overall design quality score |

## 🧪 Testing

```bash
# Run VerilogEval unit tests
pytest tests/unit/test_verilogeval_service.py

# Run integration tests
pytest tests/integration/test_verilogeval_endpoints.py
```

## 📄 License

MIT License - see LICENSE file for details.
