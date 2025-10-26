# Extracted Multi-Modal Test Data

**Total Circuits**: 15
**Source**: `test_multimodal_mermaid.py`
**Mermaid Diagrams**: ✓ Included
**Extracted**: 2025-10-23 21:08:05

---

## Directory Structure

```
extracted_multimodal/
├── INDEX.md (this file)
├── all_multimodal_data.json
├── prompts/
│   ├── all_prompts.json
│   ├── all_prompts.csv
│   └── by_complexity/
├── mermaid/
│   ├── all_diagrams.json
│   ├── by_complexity/
│   └── individual_files/
└── combined/
    └── prompt_with_mermaid/
```

## File Formats

### all_multimodal_data.json
Complete data including prompts AND Mermaid diagrams

### prompts/
- `all_prompts.json` - All prompts in JSON format
- `all_prompts.csv` - All prompts in CSV format
- `by_complexity/` - Markdown files organized by complexity

### mermaid/
- `all_diagrams.json` - All Mermaid diagrams in JSON
- `individual_files/` - Individual .mmd files for each circuit

### combined/
- `prompt_with_mermaid/` - Markdown files with both prompt and diagram

## Usage

### Load in Python

```python
import json

# Load complete multi-modal data
with open('all_multimodal_data.json', 'r') as f:
    data = json.load(f)

for circuit in data['circuits']:
    print(circuit['name'])
    print(circuit['prompt'])
    print(circuit['mermaid_code'])
```
