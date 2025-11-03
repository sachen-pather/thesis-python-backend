# Enhanced Verilog Generation and dual verification system

A comprehensive AI-powered hardware design pipeline with industry-standard VerilogEval quality assessment.

##  Quick Start

### Prerequisites

- Python 3.9+
- Node.js 16+ (for frontend)
- iverilog (for simulation)

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

## 🧪 Testing

```bash
# Run VerilogEval unit tests
pytest tests/unit/test_verilogeval_service.py

# Run integration tests
pytest tests/integration/test_verilogeval_endpoints.py
```

## 📄 License

MIT License - see LICENSE file for details.
