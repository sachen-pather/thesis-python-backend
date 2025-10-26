#!/usr/bin/env python3
"""
Prompt Extraction Script
Extracts all circuit generation prompts from test_multimodal_mermaid.py
and organizes them into structured files for easy reference and reuse.

Output structure:
extracted_prompts/
  ├── prompts_by_complexity/
  │   ├── simple_prompts.md
  │   ├── medium_prompts.md
  │   └── complex_prompts.md
  ├── prompts_by_category/
  │   ├── combinational_prompts.md
  │   ├── sequential_prompts.md
  │   ├── arithmetic_prompts.md
  │   └── state_machine_prompts.md
  ├── all_prompts.json
  ├── all_prompts.csv
  └── INDEX.md

Usage:
    # Run from root/tests/analysis/ directory:
    python extract_prompts.py

    # Or with custom paths:
    python extract_prompts.py --input /path/to/test_multimodal_mermaid.py --output /path/to/output
"""

import os
import re
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

def extract_prompts_from_file(file_path: Path) -> List[Tuple[str, str, str, str]]:
    """
    Extract circuit prompts from test_multimodal_mermaid.py
    Returns: List of (name, prompt, category, complexity) tuples
    """
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the get_test_circuits function
    match = re.search(r'def get_test_circuits\(\).*?return \[(.*?)\n    \]', content, re.DOTALL)
    
    if not match:
        print("❌ Could not find get_test_circuits() function")
        return []
    
    circuits_text = match.group(1)
    
    # Parse each circuit tuple
    circuits = []
    # Pattern: ("name", "prompt", "category", "complexity")
    pattern = r'\("([^"]+)",\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\)'
    
    for match in re.finditer(pattern, circuits_text):
        name, prompt, category, complexity = match.groups()
        circuits.append((name, prompt, category, complexity))
    
    return circuits

def create_markdown_by_complexity(circuits: List[Tuple], output_dir: Path):
    """Create separate markdown files for each complexity level"""
    
    complexity_dir = output_dir / "prompts_by_complexity"
    complexity_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by complexity
    by_complexity = {}
    for name, prompt, category, complexity in circuits:
        if complexity not in by_complexity:
            by_complexity[complexity] = []
        by_complexity[complexity].append((name, prompt, category))
    
    # Create files for each complexity
    for complexity, items in by_complexity.items():
        filename = complexity_dir / f"{complexity}_prompts.md"
        
        with open(filename, 'w') as f:
            f.write(f"# {complexity.upper()} Complexity Circuit Prompts\n\n")
            f.write(f"Total prompts: {len(items)}\n\n")
            f.write("---\n\n")
            
            for i, (name, prompt, category) in enumerate(items, 1):
                f.write(f"## {i}. {name}\n\n")
                f.write(f"**Category**: {category}  \n")
                f.write(f"**Complexity**: {complexity}\n\n")
                f.write(f"**Prompt**:\n```\n{prompt}\n```\n\n")
                f.write("---\n\n")
        
        print(f"  ✓ Created: {filename}")

def create_markdown_by_category(circuits: List[Tuple], output_dir: Path):
    """Create separate markdown files for each category"""
    
    category_dir = output_dir / "prompts_by_category"
    category_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by category
    by_category = {}
    for name, prompt, category, complexity in circuits:
        if category not in by_category:
            by_category[category] = []
        by_category[category].append((name, prompt, complexity))
    
    # Create files for each category
    for category, items in by_category.items():
        # Sanitize filename
        safe_category = category.lower().replace(' ', '_').replace('/', '_')
        filename = category_dir / f"{safe_category}_prompts.md"
        
        with open(filename, 'w') as f:
            f.write(f"# {category} Circuit Prompts\n\n")
            f.write(f"Total prompts: {len(items)}\n\n")
            f.write("---\n\n")
            
            for i, (name, prompt, complexity) in enumerate(items, 1):
                f.write(f"## {i}. {name}\n\n")
                f.write(f"**Category**: {category}  \n")
                f.write(f"**Complexity**: {complexity}\n\n")
                f.write(f"**Prompt**:\n```\n{prompt}\n```\n\n")
                f.write("---\n\n")
        
        print(f"  ✓ Created: {filename}")

def create_json_export(circuits: List[Tuple], output_dir: Path):
    """Create JSON export of all prompts"""
    
    json_data = {
        'metadata': {
            'total_prompts': len(circuits),
            'extracted_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source': 'test_multimodal_mermaid.py'
        },
        'prompts': []
    }
    
    for i, (name, prompt, category, complexity) in enumerate(circuits, 1):
        json_data['prompts'].append({
            'id': i,
            'name': name,
            'prompt': prompt,
            'category': category,
            'complexity': complexity
        })
    
    json_path = output_dir / "all_prompts.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"  ✓ Created: {json_path}")

def create_csv_export(circuits: List[Tuple], output_dir: Path):
    """Create CSV export of all prompts"""
    
    data = []
    for i, (name, prompt, category, complexity) in enumerate(circuits, 1):
        data.append({
            'ID': i,
            'Circuit Name': name,
            'Prompt': prompt,
            'Category': category,
            'Complexity': complexity,
            'Prompt Length': len(prompt)
        })
    
    df = pd.DataFrame(data)
    csv_path = output_dir / "all_prompts.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"  ✓ Created: {csv_path}")

def create_index(circuits: List[Tuple], output_dir: Path):
    """Create master index file"""
    
    index_path = output_dir / "INDEX.md"
    
    # Count by complexity and category
    complexity_counts = {}
    category_counts = {}
    
    for name, prompt, category, complexity in circuits:
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
    
    with open(index_path, 'w') as f:
        f.write("# Extracted Circuit Generation Prompts\n\n")
        f.write(f"**Total Prompts**: {len(circuits)}\n\n")
        f.write(f"**Source**: `test_multimodal_mermaid.py`\n\n")
        f.write(f"**Extracted**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("---\n\n")
        
        f.write("## Summary by Complexity\n\n")
        for complexity in ['simple', 'medium', 'complex']:
            count = complexity_counts.get(complexity, 0)
            f.write(f"- **{complexity.capitalize()}**: {count} prompts\n")
        
        f.write("\n## Summary by Category\n\n")
        for category, count in sorted(category_counts.items()):
            f.write(f"- **{category}**: {count} prompts\n")
        
        f.write("\n---\n\n")
        f.write("## Directory Structure\n\n")
        f.write("```\n")
        f.write("extracted_prompts/\n")
        f.write("├── INDEX.md (this file)\n")
        f.write("├── all_prompts.json\n")
        f.write("├── all_prompts.csv\n")
        f.write("├── prompts_by_complexity/\n")
        for complexity in sorted(complexity_counts.keys()):
            f.write(f"│   ├── {complexity}_prompts.md\n")
        f.write("└── prompts_by_category/\n")
        for category in sorted(category_counts.keys()):
            safe_name = category.lower().replace(' ', '_').replace('/', '_')
            f.write(f"    ├── {safe_name}_prompts.md\n")
        f.write("```\n\n")
        
        f.write("## All Prompts\n\n")
        
        current_complexity = None
        for i, (name, prompt, category, complexity) in enumerate(circuits, 1):
            if complexity != current_complexity:
                f.write(f"\n### {complexity.capitalize()} Circuits\n\n")
                current_complexity = complexity
            
            f.write(f"{i}. **{name}** ({category})\n")
        
        f.write("\n---\n\n")
        f.write("## Usage\n\n")
        f.write("### Access Prompts\n\n")
        f.write("- **By Complexity**: See `prompts_by_complexity/`\n")
        f.write("- **By Category**: See `prompts_by_category/`\n")
        f.write("- **JSON Format**: `all_prompts.json`\n")
        f.write("- **CSV Format**: `all_prompts.csv`\n\n")
        
        f.write("### Example: Load prompts in Python\n\n")
        f.write("```python\n")
        f.write("import json\n\n")
        f.write("with open('all_prompts.json', 'r') as f:\n")
        f.write("    data = json.load(f)\n\n")
        f.write("for prompt_data in data['prompts']:\n")
        f.write("    print(f\"{prompt_data['name']}: {prompt_data['prompt']}\")\n")
        f.write("```\n\n")
        
        f.write("### Example: Load prompts with pandas\n\n")
        f.write("```python\n")
        f.write("import pandas as pd\n\n")
        f.write("df = pd.read_csv('all_prompts.csv')\n")
        f.write("simple_prompts = df[df['Complexity'] == 'simple']\n")
        f.write("```\n")
    
    print(f"  ✓ Created: {index_path}")

def create_readme(circuits: List[Tuple], output_dir: Path):
    """Create README with detailed information"""
    
    readme_path = output_dir / "README.md"
    
    with open(readme_path, 'w') as f:
        f.write("# Circuit Generation Prompts - Extracted from Test Suite\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"This directory contains **{len(circuits)} circuit generation prompts** ")
        f.write("extracted from the multi-modal test suite (`test_multimodal_mermaid.py`).\n\n")
        
        f.write("These prompts are used to test different approaches to Verilog code generation:\n")
        f.write("1. **Prompt Only**: Direct prompt → Verilog\n")
        f.write("2. **Mermaid Only**: Prompt → Mermaid → Verilog\n")
        f.write("3. **Combined**: Prompt → Mermaid → (Prompt + Mermaid) → Verilog\n\n")
        
        f.write("## Complexity Levels\n\n")
        
        complexity_counts = {}
        for _, _, _, complexity in circuits:
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        f.write("### Simple ({} prompts)\n".format(complexity_counts.get('simple', 0)))
        f.write("Basic circuits with straightforward specifications:\n")
        for name, _, _, complexity in circuits:
            if complexity == 'simple':
                f.write(f"- {name}\n")
        f.write("\n")
        
        f.write("### Medium ({} prompts)\n".format(complexity_counts.get('medium', 0)))
        f.write("Intermediate circuits with multiple components:\n")
        for name, _, _, complexity in circuits:
            if complexity == 'medium':
                f.write(f"- {name}\n")
        f.write("\n")
        
        f.write("### Complex ({} prompts)\n".format(complexity_counts.get('complex', 0)))
        f.write("Advanced circuits with state machines and protocols:\n")
        for name, _, _, complexity in circuits:
            if complexity == 'complex':
                f.write(f"- {name}\n")
        f.write("\n")
        
        f.write("## File Formats\n\n")
        f.write("### Markdown Files\n")
        f.write("Human-readable prompts organized by complexity or category.\n")
        f.write("- Easy to browse and reference\n")
        f.write("- Good for documentation\n\n")
        
        f.write("### JSON File (`all_prompts.json`)\n")
        f.write("Machine-readable format with structured data.\n")
        f.write("- Easy to parse programmatically\n")
        f.write("- Includes metadata\n")
        f.write("- Good for automation\n\n")
        
        f.write("### CSV File (`all_prompts.csv`)\n")
        f.write("Spreadsheet-compatible format.\n")
        f.write("- Easy to analyze with Excel/Google Sheets\n")
        f.write("- Good for data analysis\n")
        f.write("- Includes prompt length metrics\n\n")
        
        f.write("## Categories\n\n")
        
        category_counts = {}
        for _, _, category, _ in circuits:
            category_counts[category] = category_counts.get(category, 0) + 1
        
        for category, count in sorted(category_counts.items()):
            f.write(f"- **{category}**: {count} prompts\n")
        
        f.write("\n## Typical Prompt Structure\n\n")
        f.write("Each prompt includes:\n")
        f.write("1. **Circuit description**: What the circuit should do\n")
        f.write("2. **Input/Output specification**: Required ports\n")
        f.write("3. **Functional requirements**: Specific behaviors\n")
        f.write("4. **Testbench requirement**: Must include verification\n\n")
        
        f.write("## Use Cases\n\n")
        f.write("- **Testing LLM code generation**: Use as inputs to GPT-4, Claude, etc.\n")
        f.write("- **Benchmarking**: Compare different generation approaches\n")
        f.write("- **Research**: Study prompt engineering for hardware design\n")
        f.write("- **Educational**: Learn how to write good hardware specifications\n")
        f.write("- **Dataset creation**: Build training data for ML models\n\n")
        
        f.write("## Example Usage\n\n")
        f.write("### Python\n")
        f.write("```python\n")
        f.write("import json\n\n")
        f.write("# Load all prompts\n")
        f.write("with open('all_prompts.json', 'r') as f:\n")
        f.write("    data = json.load(f)\n\n")
        f.write("# Get simple complexity prompts\n")
        f.write("simple_prompts = [p for p in data['prompts'] if p['complexity'] == 'simple']\n\n")
        f.write("# Use in your LLM API call\n")
        f.write("for prompt_data in simple_prompts:\n")
        f.write("    response = your_llm_api.generate(\n")
        f.write("        prompt=prompt_data['prompt'],\n")
        f.write("        task='verilog_generation'\n")
        f.write("    )\n")
        f.write("```\n\n")
        
        f.write("### JavaScript\n")
        f.write("```javascript\n")
        f.write("const fs = require('fs');\n\n")
        f.write("// Load prompts\n")
        f.write("const data = JSON.parse(fs.readFileSync('all_prompts.json', 'utf8'));\n\n")
        f.write("// Filter by category\n")
        f.write("const stateMachinePrompts = data.prompts.filter(\n")
        f.write("  p => p.category === 'State Machine'\n")
        f.write(");\n")
        f.write("```\n\n")
        
        f.write("## Prompt Statistics\n\n")
        
        # Calculate statistics
        prompt_lengths = [len(prompt) for _, prompt, _, _ in circuits]
        avg_length = sum(prompt_lengths) / len(prompt_lengths)
        min_length = min(prompt_lengths)
        max_length = max(prompt_lengths)
        
        f.write(f"- **Total Prompts**: {len(circuits)}\n")
        f.write(f"- **Average Length**: {avg_length:.0f} characters\n")
        f.write(f"- **Shortest Prompt**: {min_length} characters\n")
        f.write(f"- **Longest Prompt**: {max_length} characters\n\n")
        
        f.write("## Related Files\n\n")
        f.write("- **Source**: `tests/analysis/test_multimodal_mermaid.py`\n")
        f.write("- **Test Results**: `thesis_generation_results/multimodal_mermaid/`\n")
        f.write("- **Extracted Circuits**: `extracted_circuits/`\n")
    
    print(f"  ✓ Created: {readme_path}")

def find_project_root():
    """Find project root by looking for tests/ directory"""
    current = Path.cwd()
    
    # Check if we're in tests/analysis
    if current.name == 'analysis' and current.parent.name == 'tests':
        return current.parent.parent
    elif current.name == 'tests':
        return current.parent
    
    # Search upward
    for parent in [current] + list(current.parents):
        if (parent / 'tests').exists():
            return parent
    
    return None

def main():
    parser = argparse.ArgumentParser(
        description='Extract prompts from test_multimodal_mermaid.py',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', type=str, 
                       help='Path to test_multimodal_mermaid.py (auto-detected if not specified)')
    parser.add_argument('--output', type=str, default='extracted_prompts',
                       help='Output directory (default: extracted_prompts)')
    parser.add_argument('--root', type=str,
                       help='Project root directory (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PROMPT EXTRACTION TOOL")
    print("="*80)
    
    # Find project root
    if args.root:
        project_root = Path(args.root)
    else:
        project_root = find_project_root()
    
    if not project_root:
        print("❌ Could not find project root!")
        print("Please run from tests/analysis/ or specify --root")
        sys.exit(1)
    
    print(f"Project root: {project_root}")
    
    # Determine input file
    if args.input:
        input_file = Path(args.input)
    else:
        input_file = project_root / 'tests' / 'analysis' / 'test_multimodal_mermaid.py'
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print("Please specify --input /path/to/test_multimodal_mermaid.py")
        sys.exit(1)
    
    print(f"Input file: {input_file}")
    
    # Determine output directory
    if os.path.isabs(args.output):
        output_dir = Path(args.output)
    else:
        output_dir = project_root / args.output
    
    print(f"Output directory: {output_dir}")
    
    # Extract prompts
    print("\nExtracting prompts...")
    circuits = extract_prompts_from_file(input_file)
    
    if not circuits:
        print("❌ No prompts found!")
        sys.exit(1)
    
    print(f"✅ Found {len(circuits)} prompts")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all outputs
    print("\nGenerating output files...")
    
    print("\n📁 Creating markdown files by complexity...")
    create_markdown_by_complexity(circuits, output_dir)
    
    print("\n📁 Creating markdown files by category...")
    create_markdown_by_category(circuits, output_dir)
    
    print("\n📄 Creating JSON export...")
    create_json_export(circuits, output_dir)
    
    print("\n📊 Creating CSV export...")
    create_csv_export(circuits, output_dir)
    
    print("\n📋 Creating index...")
    create_index(circuits, output_dir)
    
    print("\n📖 Creating README...")
    create_readme(circuits, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE!")
    print("="*80)
    
    print(f"\n📊 Statistics:")
    print(f"  Total Prompts: {len(circuits)}")
    
    complexity_counts = {}
    category_counts = {}
    for _, _, category, complexity in circuits:
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
    
    print(f"\n  By Complexity:")
    for complexity in ['simple', 'medium', 'complex']:
        count = complexity_counts.get(complexity, 0)
        print(f"    {complexity.capitalize()}: {count}")
    
    print(f"\n  By Category:")
    for category, count in sorted(category_counts.items()):
        print(f"    {category}: {count}")
    
    print(f"\n📁 Files created in: {output_dir}")
    print(f"  - INDEX.md")
    print(f"  - README.md")
    print(f"  - all_prompts.json")
    print(f"  - all_prompts.csv")
    print(f"  - prompts_by_complexity/ ({len(complexity_counts)} files)")
    print(f"  - prompts_by_category/ ({len(category_counts)} files)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()