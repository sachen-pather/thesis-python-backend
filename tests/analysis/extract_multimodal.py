#!/usr/bin/env python3
"""
Complete Multi-Modal Extraction Script
Extracts prompts AND generates corresponding Mermaid diagrams

This script:
1. Extracts all 15 circuit generation prompts
2. Generates Mermaid diagrams for each prompt (via API)
3. Saves both prompts and diagrams in multiple formats

Output structure:
extracted_multimodal/
  ├── prompts/
  │   ├── all_prompts.json
  │   ├── all_prompts.csv
  │   └── by_complexity/
  ├── mermaid/
  │   ├── all_diagrams.json
  │   ├── by_complexity/
  │   └── individual_files/
  ├── combined/
  │   ├── prompt_with_mermaid/
  │   └── all_combined.json
  └── INDEX.md

Usage:
    # Basic (requires backend running):
    python extract_multimodal.py

    # Prompts only (no Mermaid generation):
    python extract_multimodal.py --prompts-only

    # Custom paths:
    python extract_multimodal.py --input /path/to/test.py --output /path/to/output
"""

import os
import re
import sys
import json
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

BASE_URL = "http://localhost:8000"

def extract_prompts_from_file(file_path: Path) -> List[Tuple[str, str, str, str]]:
    """Extract circuit prompts from test file"""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    match = re.search(r'def get_test_circuits\(\).*?return \[(.*?)\n    \]', content, re.DOTALL)
    
    if not match:
        print("❌ Could not find get_test_circuits() function")
        return []
    
    circuits_text = match.group(1)
    circuits = []
    pattern = r'\("([^"]+)",\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\)'
    
    for match in re.finditer(pattern, circuits_text):
        name, prompt, category, complexity = match.groups()
        circuits.append((name, prompt, category, complexity))
    
    return circuits

def check_backend() -> bool:
    """Check if backend is running"""
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def generate_mermaid(prompt: str, model: str = "claude") -> Tuple[bool, str, str]:
    """Generate Mermaid diagram from prompt via API"""
    try:
        response = requests.post(
            f"{BASE_URL}/api/design/generate-mermaid",
            json={"prompt": prompt, "model": model, "session_id": f"extract_{int(time.time())}"},
            timeout=90
        )
        
        if response.status_code != 200:
            return False, "", "API request failed"
        
        data = response.json()
        if not data.get("success"):
            return False, "", data.get("error", "Unknown error")
        
        mermaid_code = data.get("mermaid_code", "")
        return True, mermaid_code, ""
        
    except Exception as e:
        return False, "", str(e)

def create_prompts_only_structure(circuits: List[Tuple], output_dir: Path):
    """Create prompt extraction without Mermaid generation"""
    
    prompts_dir = output_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON export
    json_data = {
        'metadata': {
            'total_prompts': len(circuits),
            'extracted_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source': 'test_multimodal_mermaid.py',
            'mermaid_included': False
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
    
    json_path = prompts_dir / "all_prompts.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"  ✓ Created: {json_path}")
    
    # CSV export
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
    csv_path = prompts_dir / "all_prompts.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"  ✓ Created: {csv_path}")
    
    # By complexity
    by_complexity_dir = prompts_dir / "by_complexity"
    by_complexity_dir.mkdir(exist_ok=True)
    
    by_complexity = {}
    for name, prompt, category, complexity in circuits:
        if complexity not in by_complexity:
            by_complexity[complexity] = []
        by_complexity[complexity].append((name, prompt, category))
    
    for complexity, items in by_complexity.items():
        filename = by_complexity_dir / f"{complexity}_prompts.md"
        
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

def create_full_multimodal_extraction(circuits: List[Tuple], output_dir: Path, model: str = "claude"):
    """Create full extraction with Mermaid diagram generation"""
    
    print("\n" + "="*80)
    print("GENERATING MERMAID DIAGRAMS")
    print("="*80)
    print(f"Using model: {model}")
    print(f"Total circuits: {len(circuits)}")
    print(f"Estimated time: ~{len(circuits) * 20 / 60:.1f} minutes")
    print()
    
    # Create directories
    prompts_dir = output_dir / "prompts"
    mermaid_dir = output_dir / "mermaid"
    combined_dir = output_dir / "combined"
    
    prompts_dir.mkdir(parents=True, exist_ok=True)
    mermaid_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    # Data structures
    all_data = {
        'metadata': {
            'total_circuits': len(circuits),
            'extracted_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source': 'test_multimodal_mermaid.py',
            'mermaid_model': model,
            'mermaid_included': True
        },
        'circuits': []
    }
    
    # Generate Mermaid for each circuit
    for i, (name, prompt, category, complexity) in enumerate(circuits, 1):
        print(f"[{i}/{len(circuits)}] {name}...")
        
        success, mermaid_code, error = generate_mermaid(prompt, model)
        
        circuit_data = {
            'id': i,
            'name': name,
            'prompt': prompt,
            'category': category,
            'complexity': complexity,
            'mermaid_generation_success': success,
            'mermaid_code': mermaid_code if success else None,
            'mermaid_error': error if not success else None
        }
        
        all_data['circuits'].append(circuit_data)
        
        if success:
            print(f"  ✓ Generated Mermaid ({len(mermaid_code)} chars)")
            
            # Save individual Mermaid file
            safe_name = name.lower().replace(' ', '_').replace(':', '').replace('-', '_')
            mermaid_file = mermaid_dir / "individual_files" / f"{safe_name}.mmd"
            mermaid_file.parent.mkdir(exist_ok=True)
            
            with open(mermaid_file, 'w') as f:
                f.write(f"```mermaid\n{mermaid_code}\n```\n")
            
            # Save combined prompt + mermaid
            combined_file = combined_dir / "prompt_with_mermaid" / f"{safe_name}.md"
            combined_file.parent.mkdir(exist_ok=True)
            
            with open(combined_file, 'w') as f:
                f.write(f"# {name}\n\n")
                f.write(f"**Category**: {category}  \n")
                f.write(f"**Complexity**: {complexity}\n\n")
                f.write("## Original Prompt\n\n")
                f.write(f"```\n{prompt}\n```\n\n")
                f.write("## Generated Mermaid Diagram\n\n")
                f.write(f"```mermaid\n{mermaid_code}\n```\n")
        else:
            print(f"  ✗ Failed: {error}")
        
        time.sleep(2)  # Rate limiting
    
    # Save combined JSON
    json_path = output_dir / "all_multimodal_data.json"
    with open(json_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"\n✓ Saved: {json_path}")
    
    # Save prompts JSON
    prompts_json = {
        'metadata': all_data['metadata'],
        'prompts': [
            {
                'id': c['id'],
                'name': c['name'],
                'prompt': c['prompt'],
                'category': c['category'],
                'complexity': c['complexity']
            }
            for c in all_data['circuits']
        ]
    }
    
    prompts_json_path = prompts_dir / "all_prompts.json"
    with open(prompts_json_path, 'w') as f:
        json.dump(prompts_json, f, indent=2)
    
    print(f"✓ Saved: {prompts_json_path}")
    
    # Save Mermaid-only JSON
    mermaid_json = {
        'metadata': all_data['metadata'],
        'diagrams': [
            {
                'id': c['id'],
                'name': c['name'],
                'complexity': c['complexity'],
                'success': c['mermaid_generation_success'],
                'mermaid_code': c['mermaid_code'],
                'error': c['mermaid_error']
            }
            for c in all_data['circuits']
        ]
    }
    
    mermaid_json_path = mermaid_dir / "all_diagrams.json"
    with open(mermaid_json_path, 'w') as f:
        json.dump(mermaid_json, f, indent=2)
    
    print(f"✓ Saved: {mermaid_json_path}")
    
    # Create summary by complexity
    success_count = sum(1 for c in all_data['circuits'] if c['mermaid_generation_success'])
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total circuits: {len(circuits)}")
    print(f"Mermaid generated: {success_count}/{len(circuits)}")
    print(f"Success rate: {success_count/len(circuits)*100:.1f}%")
    
    # By complexity
    by_complexity = {}
    for c in all_data['circuits']:
        comp = c['complexity']
        if comp not in by_complexity:
            by_complexity[comp] = {'total': 0, 'success': 0}
        by_complexity[comp]['total'] += 1
        if c['mermaid_generation_success']:
            by_complexity[comp]['success'] += 1
    
    print(f"\nBy Complexity:")
    for comp in ['simple', 'medium', 'complex']:
        if comp in by_complexity:
            stats = by_complexity[comp]
            print(f"  {comp.capitalize()}: {stats['success']}/{stats['total']}")

def create_index(circuits: List[Tuple], output_dir: Path, has_mermaid: bool):
    """Create master INDEX.md file"""
    
    index_path = output_dir / "INDEX.md"
    
    with open(index_path, 'w') as f:
        f.write("# Extracted Multi-Modal Test Data\n\n")
        f.write(f"**Total Circuits**: {len(circuits)}\n")
        f.write(f"**Source**: `test_multimodal_mermaid.py`\n")
        f.write(f"**Mermaid Diagrams**: {'✓ Included' if has_mermaid else '✗ Not generated'}\n")
        f.write(f"**Extracted**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("---\n\n")
        
        f.write("## Directory Structure\n\n")
        f.write("```\n")
        if has_mermaid:
            f.write("extracted_multimodal/\n")
            f.write("├── INDEX.md (this file)\n")
            f.write("├── all_multimodal_data.json\n")
            f.write("├── prompts/\n")
            f.write("│   ├── all_prompts.json\n")
            f.write("│   ├── all_prompts.csv\n")
            f.write("│   └── by_complexity/\n")
            f.write("├── mermaid/\n")
            f.write("│   ├── all_diagrams.json\n")
            f.write("│   ├── by_complexity/\n")
            f.write("│   └── individual_files/\n")
            f.write("└── combined/\n")
            f.write("    └── prompt_with_mermaid/\n")
        else:
            f.write("extracted_multimodal/\n")
            f.write("├── INDEX.md (this file)\n")
            f.write("└── prompts/\n")
            f.write("    ├── all_prompts.json\n")
            f.write("    ├── all_prompts.csv\n")
            f.write("    └── by_complexity/\n")
        f.write("```\n\n")
        
        f.write("## File Formats\n\n")
        f.write("### all_multimodal_data.json\n")
        if has_mermaid:
            f.write("Complete data including prompts AND Mermaid diagrams\n\n")
        else:
            f.write("Prompts only (run with backend to generate Mermaid)\n\n")
        
        f.write("### prompts/\n")
        f.write("- `all_prompts.json` - All prompts in JSON format\n")
        f.write("- `all_prompts.csv` - All prompts in CSV format\n")
        f.write("- `by_complexity/` - Markdown files organized by complexity\n\n")
        
        if has_mermaid:
            f.write("### mermaid/\n")
            f.write("- `all_diagrams.json` - All Mermaid diagrams in JSON\n")
            f.write("- `individual_files/` - Individual .mmd files for each circuit\n\n")
            
            f.write("### combined/\n")
            f.write("- `prompt_with_mermaid/` - Markdown files with both prompt and diagram\n\n")
        
        f.write("## Usage\n\n")
        f.write("### Load in Python\n\n")
        f.write("```python\n")
        f.write("import json\n\n")
        if has_mermaid:
            f.write("# Load complete multi-modal data\n")
            f.write("with open('all_multimodal_data.json', 'r') as f:\n")
            f.write("    data = json.load(f)\n\n")
            f.write("for circuit in data['circuits']:\n")
            f.write("    print(circuit['name'])\n")
            f.write("    print(circuit['prompt'])\n")
            f.write("    print(circuit['mermaid_code'])\n")
        else:
            f.write("# Load prompts\n")
            f.write("with open('prompts/all_prompts.json', 'r') as f:\n")
            f.write("    data = json.load(f)\n\n")
            f.write("for prompt in data['prompts']:\n")
            f.write("    print(prompt['name'])\n")
            f.write("    print(prompt['prompt'])\n")
        f.write("```\n")
    
    print(f"✓ Created: {index_path}")

def find_project_root():
    """Find project root"""
    current = Path.cwd()
    
    if current.name == 'analysis' and current.parent.name == 'tests':
        return current.parent.parent
    elif current.name == 'tests':
        return current.parent
    
    for parent in [current] + list(current.parents):
        if (parent / 'tests').exists():
            return parent
    
    return None

def main():
    parser = argparse.ArgumentParser(
        description='Extract prompts and optionally generate Mermaid diagrams',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', type=str,
                       help='Path to test_multimodal_mermaid.py')
    parser.add_argument('--output', type=str, default='extracted_multimodal',
                       help='Output directory (default: extracted_multimodal)')
    parser.add_argument('--root', type=str,
                       help='Project root directory')
    parser.add_argument('--prompts-only', action='store_true',
                       help='Extract prompts only (skip Mermaid generation)')
    parser.add_argument('--model', type=str, default='claude', choices=['claude', 'gpt-4o'],
                       help='Model to use for Mermaid generation (default: claude)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MULTI-MODAL EXTRACTION TOOL")
    print("="*80)
    
    # Find project root
    if args.root:
        project_root = Path(args.root)
    else:
        project_root = find_project_root()
    
    if not project_root:
        print("❌ Could not find project root!")
        sys.exit(1)
    
    print(f"Project root: {project_root}")
    
    # Determine input file
    if args.input:
        input_file = Path(args.input)
    else:
        input_file = project_root / 'tests' / 'analysis' / 'test_multimodal_mermaid.py'
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    print(f"Input file: {input_file}")
    
    # Determine output directory
    if os.path.isabs(args.output):
        output_dir = Path(args.output)
    else:
        output_dir = project_root / args.output
    
    print(f"Output directory: {output_dir}")
    
    # Extract prompts
    print("\n" + "="*80)
    print("EXTRACTING PROMPTS")
    print("="*80)
    
    circuits = extract_prompts_from_file(input_file)
    
    if not circuits:
        print("❌ No prompts found!")
        sys.exit(1)
    
    print(f"✓ Found {len(circuits)} prompts")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Decide extraction mode
    if args.prompts_only:
        print("\n📝 Mode: Prompts only (no Mermaid generation)")
        create_prompts_only_structure(circuits, output_dir)
        has_mermaid = False
    else:
        # Check if backend is running
        if not check_backend():
            print("\n⚠️  WARNING: Backend not running!")
            print("To generate Mermaid diagrams, start backend with:")
            print("  uvicorn main:app --reload")
            print("\nFalling back to prompts-only mode...")
            create_prompts_only_structure(circuits, output_dir)
            has_mermaid = False
        else:
            print(f"\n✓ Backend is running")
            print(f"🎨 Mode: Full multi-modal extraction (prompts + Mermaid)")
            create_full_multimodal_extraction(circuits, output_dir, args.model)
            has_mermaid = True
    
    # Create index
    print("\n" + "="*80)
    print("CREATING INDEX")
    print("="*80)
    create_index(circuits, output_dir, has_mermaid)
    
    # Final summary
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE!")
    print("="*80)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📊 Total circuits: {len(circuits)}")
    print(f"🎨 Mermaid diagrams: {'✓ Generated' if has_mermaid else '✗ Not generated'}")
    
    if not has_mermaid and not args.prompts_only:
        print("\n💡 TIP: Start the backend and re-run without --prompts-only to generate Mermaid diagrams")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()