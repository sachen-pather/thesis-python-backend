# Final LLM vs VAE Verification Comparison Report

**Generated:** 2025-10-15 18:23:28

## Test Overview

- **Total circuits tested:** 26
- **Total tests run:** 104
- **Test suite:** Proven 26-circuit validation suite (80.8% VAE baseline)
- **Models compared:** GPT-4o, Claude Sonnet 3.5, Llama 3.3 70B, Gemini 2.0 Flash

## Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| VAE (Baseline) | 80.8% | 100.0% | 58.3% | 73.7% |
| GPT-4o | 0.0% | 0.0% | 0.0% | 0.0% |
| Claude Sonnet 3.5 | 0.0% | 0.0% | 0.0% | 0.0% |
| Llama 3.3 70B | 0.0% | 0.0% | 0.0% | 0.0% |
| Gemini 2.0 Flash | 0.0% | 0.0% | 0.0% | 0.0% |

## Key Findings

- **Best LLM Accuracy:** GPT-4o (0.0%)
- **VAE Baseline:** 80.8% accuracy
- **VAE Precision:** 100.0%
- **VAE Speed:** ~1.50s per circuit

## Circuit Categories Tested

- Combinational - Normal
- Combinational - Buggy
- Sequential - Normal
- Sequential - Buggy
- Arithmetic - Normal
- Arithmetic - Buggy
