#!/usr/bin/env python3
"""
Circuit Extraction Script - Updated for Real Project Structure
Extracts all 86 circuits from the test suite files and saves them as individual .v files
organized by complexity level and category.

Project Structure:
root/
├── tests/
│   ├── analysis/
│   │   └── modular_complexity_comparison.py
│   └── integration/
│       ├── comprehensive_vae_test_suite.py
│       ├── extended_test_suite.py
│       └── complex_test_suite.py

Output structure:
extracted_circuits/
  ├── simple/
  ├── medium/
  └── complex/

Usage:
    # Run from root/tests/analysis/ directory:
    python extract_circuits.py

    # Or run from anywhere with custom paths:
    python extract_circuits.py --output /path/to/output

Author: Auto-generated
Date: October 23, 2025
"""

import os
import re
import sys
import argparse
from pathlib import Path

def sanitize_filename(name):
    """Convert circuit name to valid filename"""
    filename = name.lower()
    filename = re.sub(r'[^\w\s-]', '', filename)
    filename = re.sub(r'[\s]+', '_', filename)
    return filename

def extract_circuits_from_dict(circuits_dict, complexity_level, base_output_dir):
    """
    Extract circuits from a dictionary structure
    
    Args:
        circuits_dict: Dictionary with category keys and list of (name, code, is_normal) tuples
        complexity_level: 'simple', 'medium', or 'complex'
        base_output_dir: Base directory for output
    """
    
    stats = {
        'total': 0,
        'by_category': {}
    }
    
    for category, circuits in circuits_dict.items():
        # Sanitize category name for folder
        category_folder = category.lower()
        category_folder = re.sub(r'[\s]+', '_', category_folder)
        category_folder = re.sub(r'[^\w_-]', '', category_folder)
        
        # Create output directory
        output_dir = Path(base_output_dir) / complexity_level / category_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        
        category_count = 0
        
        for name, code, is_normal in circuits:
            # Create filename
            filename = sanitize_filename(name)
            status = "normal" if is_normal else "buggy"
            filepath = output_dir / f"{filename}.v"
            
            # Add header comment to the file
            header = f"""/*
 * Circuit: {name}
 * Category: {category}
 * Complexity: {complexity_level.upper()}
 * Status: {status.upper()}
 * 
 * Extracted from test suite
 */

"""
            
            # Write the circuit to file
            with open(filepath, 'w') as f:
                f.write(header)
                f.write(code)
            
            category_count += 1
            stats['total'] += 1
            
            print(f"  ✓ Created: {filepath}")
        
        stats['by_category'][category] = category_count
    
    return stats

def find_project_root():
    """
    Find the project root by looking for the tests/ directory structure
    """
    current = Path.cwd()
    
    # Check if we're already in tests/analysis or tests/integration
    if current.name == 'analysis' and (current.parent.name == 'tests'):
        return current.parent.parent
    elif current.name == 'integration' and (current.parent.name == 'tests'):
        return current.parent.parent
    elif current.name == 'tests':
        return current.parent
    
    # Search upward for tests directory
    for parent in [current] + list(current.parents):
        tests_dir = parent / 'tests'
        if tests_dir.exists() and tests_dir.is_dir():
            integration_dir = tests_dir / 'integration'
            if integration_dir.exists():
                return parent
    
    return None

def load_and_extract_simple_circuits(test_suite_path, base_output_dir):
    """Extract circuits from the simple/comprehensive test suite"""
    
    print("\n" + "="*80)
    print("EXTRACTING SIMPLE CIRCUITS")
    print("="*80)
    print(f"Source: {test_suite_path}")
    
    if not test_suite_path.exists():
        print(f"❌ ERROR: File not found: {test_suite_path}")
        return None
    
    # Add the directory to sys.path so we can import
    sys.path.insert(0, str(test_suite_path.parent))
    
    try:
        # Import the module
        from comprehensive_vae_test_suite import get_test_circuits
        
        circuits = get_test_circuits()
        stats = extract_circuits_from_dict(circuits, 'simple', base_output_dir)
        return stats
    except ImportError as e:
        print(f"❌ ERROR: Could not import: {e}")
        return None
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None

def load_and_extract_medium_circuits(test_suite_path, base_output_dir):
    """Extract circuits from the medium/extended test suite"""
    
    print("\n" + "="*80)
    print("EXTRACTING MEDIUM CIRCUITS")
    print("="*80)
    print(f"Source: {test_suite_path}")
    
    if not test_suite_path.exists():
        print(f"❌ ERROR: File not found: {test_suite_path}")
        return None
    
    # Add the directory to sys.path so we can import
    sys.path.insert(0, str(test_suite_path.parent))
    
    try:
        # Import the module
        from extended_test_suite import get_extended_test_circuits
        
        circuits = get_extended_test_circuits()
        stats = extract_circuits_from_dict(circuits, 'medium', base_output_dir)
        return stats
    except ImportError as e:
        print(f"❌ ERROR: Could not import: {e}")
        return None
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None

def load_and_extract_complex_circuits(test_suite_path, base_output_dir):
    """Extract circuits from the complex test suite"""
    
    print("\n" + "="*80)
    print("EXTRACTING COMPLEX CIRCUITS")
    print("="*80)
    print(f"Source: {test_suite_path}")
    
    if not test_suite_path.exists():
        print(f"❌ ERROR: File not found: {test_suite_path}")
        return None
    
    # Add the directory to sys.path so we can import
    sys.path.insert(0, str(test_suite_path.parent))
    
    try:
        # Import the module
        from complex_test_suite import get_complex_test_circuits
        
        circuits = get_complex_test_circuits()
        stats = extract_circuits_from_dict(circuits, 'complex', base_output_dir)
        return stats
    except ImportError as e:
        print(f"❌ ERROR: Could not import: {e}")
        return None
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None

def create_index_file(base_output_dir, all_stats):
    """Create an index file listing all extracted circuits"""
    
    index_path = Path(base_output_dir) / "INDEX.md"
    
    with open(index_path, 'w') as f:
        f.write("# Extracted Verilog Circuits\n\n")
        f.write("This directory contains 86 Verilog circuits extracted from the test suites.\n\n")
        
        f.write("## Summary\n\n")
        
        total = 0
        for complexity, stats in all_stats.items():
            if stats:
                f.write(f"### {complexity.upper()} Circuits: {stats['total']}\n\n")
                for category, count in stats['by_category'].items():
                    f.write(f"- **{category}**: {count} circuits\n")
                f.write("\n")
                total += stats['total']
        
        f.write(f"**GRAND TOTAL: {total} circuits**\n\n")
        
        f.write("## Directory Structure\n\n")
        f.write("```\n")
        f.write("extracted_circuits/\n")
        f.write("├── simple/          (26 circuits)\n")
        f.write("├── medium/          (48 circuits)\n")
        f.write("└── complex/         (12 circuits)\n")
        f.write("```\n\n")
        
        f.write("## Usage\n\n")
        f.write("Each `.v` file contains:\n")
        f.write("- The main circuit module\n")
        f.write("- A complete testbench module\n")
        f.write("- Header comments with circuit metadata\n\n")
        
        f.write("To simulate a circuit:\n")
        f.write("```bash\n")
        f.write("iverilog -o output.vvp circuit_name.v\n")
        f.write("vvp output.vvp\n")
        f.write("gtkwave dump.vcd\n")
        f.write("```\n")
    
    print(f"\n📄 Created index file: {index_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Extract Verilog circuits from test suites',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run from tests/analysis/ directory (auto-detects paths):
  python extract_circuits.py

  # Specify custom output directory:
  python extract_circuits.py --output /path/to/output

  # Specify custom project root:
  python extract_circuits.py --root /path/to/project/root
        """
    )
    parser.add_argument('--root', type=str, help='Project root directory (auto-detected if not specified)')
    parser.add_argument('--output', type=str, default='extracted_circuits', 
                       help='Output directory name (default: extracted_circuits)')
    
    args = parser.parse_args()
    
    # Find project root
    if args.root:
        project_root = Path(args.root)
    else:
        project_root = find_project_root()
    
    if not project_root:
        print("❌ ERROR: Could not find project root!")
        print("Please run this script from tests/analysis/ or tests/integration/")
        print("Or specify --root /path/to/project/root")
        sys.exit(1)
    
    print("="*80)
    print("CIRCUIT EXTRACTION TOOL")
    print("="*80)
    print(f"Project root: {project_root}")
    
    # Define paths to test suite files
    integration_dir = project_root / 'tests' / 'integration'
    
    test_suite_files = {
        'simple': integration_dir / 'comprehensive_vae_test_suite.py',
        'medium': integration_dir / 'extended_test_suite.py',
        'complex': integration_dir / 'complex_test_suite.py'
    }
    
    # Check if files exist
    print("\nChecking test suite files:")
    all_exist = True
    for name, path in test_suite_files.items():
        if path.exists():
            print(f"  ✓ Found: {name} - {path}")
        else:
            print(f"  ✗ Missing: {name} - {path}")
            all_exist = False
    
    if not all_exist:
        print("\n❌ ERROR: Some test suite files are missing!")
        print("Expected structure:")
        print("  root/tests/integration/comprehensive_vae_test_suite.py")
        print("  root/tests/integration/extended_test_suite.py")
        print("  root/tests/integration/complex_test_suite.py")
        sys.exit(1)
    
    # Define output directory
    if os.path.isabs(args.output):
        base_output_dir = Path(args.output)
    else:
        base_output_dir = project_root / args.output
    
    print(f"\nOutput directory: {base_output_dir}")
    print(f"Total circuits to extract: 86")
    
    # Extract circuits from each file
    all_stats = {}
    
    try:
        # Extract simple circuits
        stats_simple = load_and_extract_simple_circuits(test_suite_files['simple'], base_output_dir)
        all_stats['simple'] = stats_simple
        
        # Extract medium circuits
        stats_medium = load_and_extract_medium_circuits(test_suite_files['medium'], base_output_dir)
        all_stats['medium'] = stats_medium
        
        # Extract complex circuits
        stats_complex = load_and_extract_complex_circuits(test_suite_files['complex'], base_output_dir)
        all_stats['complex'] = stats_complex
        
    except Exception as e:
        print(f"\n❌ ERROR during extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create index file
    create_index_file(base_output_dir, all_stats)
    
    # Print final summary
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE!")
    print("="*80)
    
    total = 0
    for complexity, stats in all_stats.items():
        if stats:
            print(f"\n{complexity.upper()}: {stats['total']} circuits")
            for category, count in stats['by_category'].items():
                print(f"  - {category}: {count}")
            total += stats['total']
    
    print(f"\n{'='*80}")
    print(f"TOTAL CIRCUITS EXTRACTED: {total}")
    print(f"{'='*80}")
    print(f"\n📁 All circuits saved to: {base_output_dir}")
    print(f"📄 See INDEX.md for details")

if __name__ == "__main__":
    main()