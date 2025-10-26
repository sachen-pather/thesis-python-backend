# services/verilogeval_service.py
import json
import re
import tempfile
import subprocess
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class VerilogEvalResult:
    """VerilogEval assessment result"""
    functional_score: float
    syntax_score: float
    benchmark_score: float
    passed_tests: int
    total_tests: int
    issues: List[str]
    recommendations: List[str]
    complexity_rating: str

class VerilogEvalService:
    def __init__(self):
        # HDLBits-style problem categories
        self.problem_categories = {
            'combinational': ['mux', 'decoder', 'encoder', 'adder', 'comparator'],
            'sequential': ['counter', 'shift_register', 'fsm', 'memory'],
            'advanced': ['uart', 'spi', 'alu', 'processor']
        }
        
        # VerilogEval compliance patterns
        self.compliance_patterns = {
            'module_structure': r'module\s+\w+\s*\([^)]*\);.*?endmodule',
            'testbench_present': r'module\s+\w*testbench\w*',
            'proper_sensitivity': r'always\s*@\s*\([^)]*\)',
            'non_blocking': r'<=',
            'blocking': r'(?<![<>])=(?![=<>])',
            'clock_reset': r'@\s*\(\s*posedge\s+\w+\s+or\s+negedge\s+\w+\s*\)',
            'timescale': r'`timescale\s+\d+\w+/\d+\w+',
            'dump_commands': r'\$dump\w+',
            'monitor_commands': r'\$monitor'
        }

    def evaluate_verilog_comprehensive(self, verilog_code: str, user_prompt: str = "", 
                                     mermaid_code: str = "") -> VerilogEvalResult:
        """Comprehensive VerilogEval-style evaluation"""
        
        # Initialize scores
        functional_score = 0.0
        syntax_score = 0.0
        benchmark_score = 0.0
        issues = []
        recommendations = []
        passed_tests = 0
        total_tests = 0
        
        # 1. Syntax and Structure Analysis (30 points)
        syntax_results = self._evaluate_syntax_compliance(verilog_code)
        syntax_score = syntax_results['score']
        issues.extend(syntax_results['issues'])
        recommendations.extend(syntax_results['recommendations'])
        
        # 2. Functional Correctness Simulation (40 points)
        functional_results = self._evaluate_functional_correctness(verilog_code)
        functional_score = functional_results['score']
        passed_tests = functional_results['passed_tests']
        total_tests = functional_results['total_tests']
        issues.extend(functional_results['issues'])
        
        # 3. Design Quality and Best Practices (30 points)
        quality_results = self._evaluate_design_quality(verilog_code, user_prompt, mermaid_code)
        benchmark_score = quality_results['score']
        recommendations.extend(quality_results['recommendations'])
        
        # Calculate overall complexity
        complexity_rating = self._assess_complexity(verilog_code, user_prompt)
        
        return VerilogEvalResult(
            functional_score=functional_score,
            syntax_score=syntax_score,
            benchmark_score=benchmark_score,
            passed_tests=passed_tests,
            total_tests=total_tests,
            issues=issues,
            recommendations=recommendations,
            complexity_rating=complexity_rating
        )

    def _evaluate_syntax_compliance(self, verilog_code: str) -> Dict:
        """Evaluate syntax compliance using VerilogEval patterns"""
        score = 0.0
        max_score = 30.0
        issues = []
        recommendations = []
        
        # Check module structure (5 points)
        if re.search(self.compliance_patterns['module_structure'], verilog_code, re.DOTALL):
            score += 5
        else:
            issues.append("CRITICAL: Invalid or missing module structure")
            recommendations.append("Ensure proper module...endmodule blocks")
        
        # Check testbench presence (5 points)
        if re.search(self.compliance_patterns['testbench_present'], verilog_code):
            score += 5
        else:
            issues.append("WARNING: No testbench found")
            recommendations.append("Add comprehensive testbench for verification")
        
        # Check proper sensitivity lists (5 points)
        always_blocks = re.findall(r'always\s*@[^;]+', verilog_code)
        if always_blocks:
            proper_sensitivity = all('@' in block for block in always_blocks)
            if proper_sensitivity:
                score += 5
            else:
                issues.append("CRITICAL: Improper always block sensitivity lists")
                recommendations.append("Use proper sensitivity lists: @(posedge clk)")
        
        # Check assignment types (5 points)
        has_non_blocking = bool(re.search(self.compliance_patterns['non_blocking'], verilog_code))
        has_blocking = bool(re.search(self.compliance_patterns['blocking'], verilog_code))
        
        if has_non_blocking and 'always' in verilog_code:
            score += 3
        if has_blocking and ('assign' in verilog_code or 'always' in verilog_code):
            score += 2
        
        # Check timescale directive (3 points)
        if re.search(self.compliance_patterns['timescale'], verilog_code):
            score += 3
        else:
            recommendations.append("Add `timescale directive for proper simulation")
        
        # Check simulation commands (4 points)
        if re.search(self.compliance_patterns['dump_commands'], verilog_code):
            score += 2
        if re.search(self.compliance_patterns['monitor_commands'], verilog_code):
            score += 2
        
        # Check clock and reset patterns (3 points)
        if re.search(self.compliance_patterns['clock_reset'], verilog_code):
            score += 3
        elif 'posedge' in verilog_code and 'reset' in verilog_code.lower():
            score += 1
            recommendations.append("Consider proper clock/reset sensitivity: @(posedge clk or negedge rst_n)")
        
        return {
            'score': score,
            'issues': issues,
            'recommendations': recommendations
        }

    def _evaluate_functional_correctness(self, verilog_code: str) -> Dict:
        """Simulate and test functional correctness"""
        score = 0.0
        max_score = 40.0
        issues = []
        passed_tests = 0
        total_tests = 5  # Basic test suite
        
        try:
            # Test 1: Compilation (10 points)
            if self._test_compilation(verilog_code):
                score += 10
                passed_tests += 1
            else:
                issues.append("CRITICAL: Code fails to compile")
                return {'score': 0, 'passed_tests': 0, 'total_tests': total_tests, 'issues': issues}
            
            # Test 2: Simulation runs without errors (10 points)
            if self._test_simulation_execution(verilog_code):
                score += 10
                passed_tests += 1
            else:
                issues.append("CRITICAL: Simulation fails to execute")
            
            # Test 3: Generates waveform data (8 points)
            if self._test_waveform_generation(verilog_code):
                score += 8
                passed_tests += 1
            else:
                issues.append("WARNING: No waveform data generated")
            
            # Test 4: Reset functionality (6 points)
            if self._test_reset_behavior(verilog_code):
                score += 6
                passed_tests += 1
            else:
                issues.append("WARNING: Reset behavior not verified")
            
            # Test 5: Clock edge behavior (6 points)
            if self._test_clock_behavior(verilog_code):
                score += 6
                passed_tests += 1
            else:
                issues.append("WARNING: Clock edge behavior unclear")
                
        except Exception as e:
            issues.append(f"ERROR: Functional testing failed: {str(e)}")
        
        return {
            'score': score,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'issues': issues
        }

    def _evaluate_design_quality(self, verilog_code: str, user_prompt: str, mermaid_code: str) -> Dict:
        """Evaluate design quality and best practices"""
        score = 0.0
        max_score = 30.0
        recommendations = []
        
        # Modularity and structure (10 points)
        modules = re.findall(r'module\s+(\w+)', verilog_code)
        if len(modules) >= 2:  # Design + testbench
            score += 5
            if len(modules) > 2:  # Additional modularity
                score += 3
        else:
            recommendations.append("Consider modular design with separate testbench")
        
        # Signal naming conventions (5 points)
        signals = re.findall(r'(?:input|output|reg|wire)\s+(?:\[\d+:\d+\])?\s*(\w+)', verilog_code)
        meaningful_names = sum(1 for s in signals if len(s) > 2 and not s.isdigit())
        if len(signals) > 0:
            naming_score = min(5, (meaningful_names / len(signals)) * 5)
            score += naming_score
        
        # Code organization and comments (5 points)
        lines = verilog_code.split('\n')
        comment_lines = sum(1 for line in lines if '//' in line or '/*' in line)
        if len(lines) > 0:
            comment_ratio = comment_lines / len(lines)
            if comment_ratio > 0.1:  # At least 10% comments
                score += 3
            if comment_ratio > 0.2:  # 20% or more
                score += 2
        
        # Design pattern recognition (5 points)
        patterns_found = 0
        if 'counter' in user_prompt.lower() and ('+ 1' in verilog_code or '++' in verilog_code):
            patterns_found += 1
        if 'mux' in user_prompt.lower() and 'case' in verilog_code:
            patterns_found += 1
        if 'shift' in user_prompt.lower() and ('<<' in verilog_code or '>>' in verilog_code):
            patterns_found += 1
        if 'fsm' in user_prompt.lower() and 'state' in verilog_code.lower():
            patterns_found += 1
        
        score += min(5, patterns_found * 2)
        
        # Alignment with Mermaid diagram (5 points)
        if mermaid_code:
            mermaid_signals = re.findall(r'(\w+)\[', mermaid_code)
            verilog_signals = re.findall(r'(?:input|output|reg|wire)\s+(?:\[\d+:\d+\])?\s*(\w+)', verilog_code)
            
            if mermaid_signals:
                alignment_score = len(set(mermaid_signals) & set(verilog_signals)) / len(set(mermaid_signals))
                score += min(5, alignment_score * 5)
        
        return {
            'score': score,
            'recommendations': recommendations
        }

    def _test_compilation(self, verilog_code: str) -> bool:
        """Test if code compiles successfully"""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                design_path = os.path.join(tmpdir, "test_design.v")
                with open(design_path, "w") as f:
                    f.write(verilog_code)
                
                result = subprocess.run(
                    ["iverilog", "-t", "null", design_path],
                    capture_output=True,
                    timeout=10
                )
                return result.returncode == 0
        except:
            return False

    def _test_simulation_execution(self, verilog_code: str) -> bool:
        """Test if simulation executes without runtime errors"""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                design_path = os.path.join(tmpdir, "test_design.v")
                sim_path = os.path.join(tmpdir, "sim")
                
                with open(design_path, "w") as f:
                    f.write(verilog_code)
                
                # Compile
                compile_result = subprocess.run(
                    ["iverilog", "-o", sim_path, design_path],
                    capture_output=True,
                    timeout=10
                )
                
                if compile_result.returncode != 0:
                    return False
                
                # Run simulation
                sim_result = subprocess.run(
                    ["vvp", sim_path],
                    capture_output=True,
                    timeout=10
                )
                
                return sim_result.returncode == 0
        except:
            return False

    def _test_waveform_generation(self, verilog_code: str) -> bool:
        """Test if waveform files are generated"""
        return '$dumpfile' in verilog_code and '$dumpvars' in verilog_code

    def _test_reset_behavior(self, verilog_code: str) -> bool:
        """Test if reset behavior is properly implemented"""
        has_reset = any(word in verilog_code.lower() for word in ['reset', 'rst', 'rst_n'])
        has_reset_logic = 'if' in verilog_code and has_reset
        return has_reset_logic

    def _test_clock_behavior(self, verilog_code: str) -> bool:
        """Test if clock behavior is properly implemented"""
        has_clock = any(word in verilog_code.lower() for word in ['clk', 'clock'])
        has_posedge = 'posedge' in verilog_code
        return has_clock and has_posedge

    def _assess_complexity(self, verilog_code: str, user_prompt: str) -> str:
        """Assess design complexity based on VerilogEval categories"""
        
        # Count complexity indicators
        complexity_score = 0
        
        # Lines of code
        loc = len([line for line in verilog_code.split('\n') if line.strip()])
        if loc > 100:
            complexity_score += 3
        elif loc > 50:
            complexity_score += 2
        elif loc > 20:
            complexity_score += 1
        
        # Number of modules
        modules = len(re.findall(r'module\s+\w+', verilog_code))
        complexity_score += min(modules, 3)
        
        # Always blocks
        always_blocks = len(re.findall(r'always\s*@', verilog_code))
        complexity_score += min(always_blocks, 3)
        
        # State machines
        if 'state' in verilog_code.lower() or 'fsm' in user_prompt.lower():
            complexity_score += 2
        
        # Advanced features
        advanced_features = ['case', 'for', 'while', 'generate']
        complexity_score += sum(1 for feature in advanced_features if feature in verilog_code)
        
        # Classify complexity
        if complexity_score >= 8:
            return "Advanced"
        elif complexity_score >= 4:
            return "Intermediate"
        else:
            return "Basic"

    def generate_pass_at_k_metric(self, verilog_codes: List[str], k: int = 1) -> float:
        """Calculate pass@k metric as used in VerilogEval"""
        if not verilog_codes:
            return 0.0
        
        successful_codes = 0
        for code in verilog_codes[:k]:
            if self._test_compilation(code) and self._test_simulation_execution(code):
                successful_codes += 1
        
        return successful_codes / min(k, len(verilog_codes))

    def get_benchmark_comparison(self, verilog_code: str, user_prompt: str) -> Dict:
        """Compare against VerilogEval benchmark patterns"""
        
        # Identify similar problems from VerilogEval categories
        problem_type = self._identify_problem_type(user_prompt)
        
        # Get expected patterns for this problem type
        expected_patterns = self._get_expected_patterns(problem_type)
        
        # Analyze code against patterns
        pattern_matches = 0
        total_patterns = len(expected_patterns)
        
        for pattern in expected_patterns:
            if re.search(pattern, verilog_code, re.IGNORECASE):
                pattern_matches += 1
        
        score = (pattern_matches / total_patterns) * 100 if total_patterns > 0 else 0
        
        return {
            'problem_type': problem_type,
            'pattern_match_score': score,
            'patterns_found': pattern_matches,
            'total_patterns': total_patterns,
            'similar_hdlbits_problems': self._get_similar_problems(problem_type)
        }

    def _identify_problem_type(self, user_prompt: str) -> str:
        """Identify the type of digital design problem"""
        prompt_lower = user_prompt.lower()
        
        for category, keywords in self.problem_categories.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return category
        
        return 'general'

    def _get_expected_patterns(self, problem_type: str) -> List[str]:
        """Get expected code patterns for problem type"""
        patterns = {
            'combinational': [
                r'assign\s+\w+\s*=',
                r'always\s*@\s*\(\s*\*\s*\)',
                r'case\s*\(',
            ],
            'sequential': [
                r'always\s*@\s*\(\s*posedge',
                r'<=',
                r'reset|rst',
                r'if\s*\(',
            ],
            'advanced': [
                r'parameter\s+\w+',
                r'generate\s+',
                r'for\s*\(',
                r'state\s*=',
            ]
        }
        
        return patterns.get(problem_type, [])

    def _get_similar_problems(self, problem_type: str) -> List[str]:
        """Get similar HDLBits problems for reference"""
        similar_problems = {
            'combinational': ['mux2to1', 'mux4to1', 'decoder3to8', 'priority_encoder'],
            'sequential': ['counter4bit', 'shift_reg', 'dff_with_enable', 'lfsr'],
            'advanced': ['uart_tx', 'spi_master', 'alu_4bit', 'simple_cpu']
        }
        
        return similar_problems.get(problem_type, [])