#!/bin/bash

# Complete Analysis Pipeline
# Run from tests/analysis/ directory

echo "=================================="
echo "🎯 COMPLETE ANALYSIS PIPELINE"
echo "=================================="
echo ""

# Check if results exist
if [ ! -d "../../thesis_generation_results/comprehensive" ]; then
    echo "❌ No test results found!"
    echo "   Run: python test_comprehensive_suite.py"
    exit 1
fi

echo "Step 1: Generating standard plots..."
echo "-----------------------------------"
python generate_thesis_plots.py

if [ $? -ne 0 ]; then
    echo "❌ Standard analysis failed!"
    exit 1
fi

echo ""
echo "Step 2: Running advanced statistical analysis..."
echo "------------------------------------------------"
python advanced_statistical_analysis.py

if [ $? -ne 0 ]; then
    echo "❌ Advanced analysis failed!"
    exit 1
fi

echo ""
echo "=================================="
echo "✅ ANALYSIS COMPLETE!"
echo "=================================="
echo ""
echo "📊 Generated outputs:"
echo "  - 10 visualization plots"
echo "  - Statistical analysis report"
echo "  - LaTeX table for thesis"
echo "  - Text summary report"
echo ""
echo "📁 Location: thesis_generation_results/analysis_plots/"
echo ""