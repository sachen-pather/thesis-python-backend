"""
Test Suite Validator - FIXED VERSION
Correctly distinguishes combinational vs sequential circuits

SAVE AS: tests/validation/validate_test_suite.py
RUN: python tests/validation/validate_test_suite.py

KEY FIX: Only edge-triggered always blocks are sequential!
- always @(posedge clk) = SEQUENTIAL (needs clock)
- always @(*) with output reg = COMBINATIONAL (no clock needed)
"""

import re
import os
import sys
import subprocess
import tempfile
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Import the test suite to validate
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@dataclass
class ValidationResult:
    circuit_name: str
    category: str
    passed: bool
    issues: List[str]
    warnings: List[str]
    
    # Compilation checks
    compiles: bool
    simulation_runs: bool
    generates_waveform: bool
    
    # Structure checks
    has_timescale: bool
    has_testbench: bool
    has_dumpfile: bool
    has_dumpvars: bool
    has_monitor: bool
    has_finish: bool
    
    # Sequential-specific checks
    is_sequential: bool
    has_clock: bool
    clock_in_dumpvars: bool
    has_reset: bool
    
    # Testbench quality
    num_test_vectors: int
    signal_transitions: int
    waveform_signals: List[str]
    
    # Expected behavior
    expected_normal: bool
    bug_type: Optional[str]

def log(message, status="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    emoji = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️", "TEST": "🧪"}.get(status, "📝")
    print(f"[{timestamp}] {emoji} {message}")

class CircuitValidator:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def validate_circuit(self, name: str, verilog_code: str, expected_normal: bool, 
                        category: str) -> ValidationResult:
        """Comprehensive validation of a single circuit"""
        
        issues = []
        warnings = []
        
        # Initialize result
        result = ValidationResult(
            circuit_name=name,
            category=category,
            passed=False,
            issues=[],
            warnings=[],
            compiles=False,
            simulation_runs=False,
            generates_waveform=False,
            has_timescale=False,
            has_testbench=False,
            has_dumpfile=False,
            has_dumpvars=False,
            has_monitor=False,
            has_finish=False,
            is_sequential=False,
            has_clock=False,
            clock_in_dumpvars=False,
            has_reset=False,
            num_test_vectors=0,
            signal_transitions=0,
            waveform_signals=[],
            expected_normal=expected_normal,
            bug_type=None
        )
        
        # ========================================
        # 1. STRUCTURE VALIDATION
        # ========================================
        
        # Check for required elements
        result.has_timescale = '`timescale' in verilog_code
        if not result.has_timescale:
            issues.append("Missing `timescale directive")
        
        result.has_testbench = 'module testbench' in verilog_code.lower()
        if not result.has_testbench:
            issues.append("Missing testbench module")
        
        result.has_dumpfile = '$dumpfile' in verilog_code
        if not result.has_dumpfile:
            issues.append("Missing $dumpfile statement")
        
        result.has_dumpvars = '$dumpvars' in verilog_code
        if not result.has_dumpvars:
            issues.append("Missing $dumpvars statement")
        
        result.has_monitor = '$monitor' in verilog_code
        if not result.has_monitor:
            warnings.append("Missing $monitor statement (non-critical)")
        
        result.has_finish = '$finish' in verilog_code
        if not result.has_finish:
            issues.append("Missing $finish statement")
        
        # ========================================
        # 2. SEQUENTIAL CIRCUIT CHECKS - FIXED!
        # ========================================
        
        # CRITICAL FIX: Only edge-triggered always blocks are sequential!
        # always @(*) with output reg is COMBINATIONAL, not sequential
        # Only check for posedge/negedge, NOT for 'output reg'
        result.is_sequential = any(keyword in verilog_code for keyword in [
            'always @(posedge', 'always @(negedge', 
            'always @ (posedge', 'always @ (negedge'
        ])
        
        if result.is_sequential:
            # Check for clock
            result.has_clock = bool(re.search(r'(input\s+(wire\s+)?clk|reg\s+clk)', verilog_code))
            if not result.has_clock:
                issues.append("Sequential circuit missing clock signal")
            
            # Check for clock generation
            has_clock_gen = 'forever' in verilog_code and '#5 clk' in verilog_code
            if not has_clock_gen:
                issues.append("Sequential circuit missing clock generation (forever #5 clk)")
            
            # Check for reset
            result.has_reset = bool(re.search(r'(input\s+(wire\s+)?rst|reg\s+rst)', verilog_code))
            if not result.has_reset:
                warnings.append("Sequential circuit without reset signal")
            
            # CRITICAL: Check if clock is dumped
            # Look for dumpvars(0, testbench) which dumps everything
            if '$dumpvars(0, testbench)' in verilog_code:
                result.clock_in_dumpvars = True
            else:
                # Check if clock explicitly mentioned in dumpvars
                dumpvars_match = re.search(r'\$dumpvars\([^)]+\)', verilog_code)
                if dumpvars_match and 'clk' in dumpvars_match.group():
                    result.clock_in_dumpvars = True
                else:
                    issues.append("Clock signal not included in $dumpvars - use $dumpvars(0, testbench)")
        
        # ========================================
        # 3. TEST VECTOR QUALITY
        # ========================================
        
        # Count test vectors (transitions with #)
        transitions = re.findall(r'#\d+', verilog_code)
        result.num_test_vectors = len(transitions)
        
        if result.num_test_vectors < 4:
            warnings.append(f"Only {result.num_test_vectors} test vectors (recommend 4+)")
        
        # Count signal assignments in testbench
        testbench_section = verilog_code.split('module testbench')[-1] if 'module testbench' in verilog_code else ""
        signal_changes = len(re.findall(r'\w+\s*=\s*[^;]+;', testbench_section))
        result.signal_transitions = signal_changes
        
        if result.signal_transitions < 8:
            warnings.append(f"Only {result.signal_transitions} signal changes (recommend 8+)")
        
        # ========================================
        # 4. COMPILATION & SIMULATION
        # ========================================
        
        compile_result = self._compile_and_simulate(name, verilog_code)
        result.compiles = compile_result['compiles']
        result.simulation_runs = compile_result['simulation_runs']
        result.generates_waveform = compile_result['generates_waveform']
        result.waveform_signals = compile_result['waveform_signals']
        
        if not result.compiles:
            issues.append(f"Compilation failed: {compile_result['error']}")
        elif not result.simulation_runs:
            issues.append(f"Simulation failed: {compile_result['error']}")
        elif not result.generates_waveform:
            issues.append("Simulation didn't generate waveform data")
        
        # Check waveform has expected signals (only for sequential circuits)
        if result.is_sequential and result.generates_waveform:
            if 'clk' not in result.waveform_signals:
                issues.append("Clock signal not present in waveform output")
        
        # ========================================
        # 5. BUG TYPE DETECTION (for buggy circuits)
        # ========================================
        
        if not expected_normal:
            result.bug_type = self._detect_bug_type(name, verilog_code)
            if not result.bug_type:
                warnings.append("Could not identify bug type from name/code")
        
        # ========================================
        # 6. FINAL VERDICT
        # ========================================
        
        result.issues = issues
        result.warnings = warnings
        result.passed = (len(issues) == 0 and result.compiles and result.simulation_runs)
        
        return result
    
    def _compile_and_simulate(self, name: str, verilog_code: str) -> Dict:
        """Compile and simulate the circuit"""
        result = {
            'compiles': False,
            'simulation_runs': False,
            'generates_waveform': False,
            'waveform_signals': [],
            'error': None
        }
        
        # Write verilog to temp file - sanitize filename
        safe_name = re.sub(r'[^\w\s-]', '_', name).replace(' ', '_')
        verilog_file = os.path.join(self.temp_dir, f"{safe_name}.v")
        try:
            with open(verilog_file, 'w') as f:
                f.write(verilog_code)
        except Exception as e:
            result['error'] = f"Failed to write file: {e}"
            return result
        
        # Compile with iverilog
        output_file = os.path.join(self.temp_dir, f"{safe_name}.vvp")
        try:
            compile_process = subprocess.run(
                ['iverilog', '-o', output_file, verilog_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if compile_process.returncode != 0:
                result['error'] = compile_process.stderr
                return result
            
            result['compiles'] = True
            
        except subprocess.TimeoutExpired:
            result['error'] = "Compilation timeout"
            return result
        except FileNotFoundError:
            result['error'] = "iverilog not found"
            return result
        except Exception as e:
            result['error'] = f"Compilation error: {e}"
            return result
        
        # Run simulation
        try:
            sim_process = subprocess.run(
                ['vvp', output_file],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.temp_dir
            )
            
            if sim_process.returncode != 0:
                result['error'] = sim_process.stderr
                return result
            
            result['simulation_runs'] = True
            
            # Check if VCD was generated
            vcd_file = os.path.join(self.temp_dir, 'dump.vcd')
            if os.path.exists(vcd_file):
                result['generates_waveform'] = True
                
                # Parse VCD to get signal list
                with open(vcd_file, 'r') as f:
                    vcd_content = f.read()
                    signals = re.findall(r'\$var\s+\w+\s+\d+\s+\S+\s+(\w+)', vcd_content)
                    result['waveform_signals'] = list(set(signals))
            
        except subprocess.TimeoutExpired:
            result['error'] = "Simulation timeout"
            return result
        except Exception as e:
            result['error'] = f"Simulation error: {e}"
            return result
        
        return result
    
    def _detect_bug_type(self, name: str, code: str) -> Optional[str]:
        """Detect what type of bug is in the circuit"""
        name_lower = name.lower()
        
        if 'stuck' in name_lower or 'always 0' in name_lower or 'always 1' in name_lower:
            return 'stuck_signal'
        elif 'inverted' in name_lower or 'complement' in name_lower:
            return 'inverted_logic'
        elif 'wrong' in name_lower or 'acts like' in name_lower:
            return 'wrong_operator'
        elif 'no reset' in name_lower or 'missing reset' in name_lower:
            return 'missing_reset'
        elif 'no shift' in name_lower or 'ignores' in name_lower:
            return 'missing_logic'
        elif 'no carry' in name_lower or 'no borrow' in name_lower or 'no underflow' in name_lower:
            return 'missing_output'
        elif 'off by' in name_lower or 'wrong limit' in name_lower or 'wrong initialization' in name_lower:
            return 'off_by_one'
        elif 'broken' in name_lower or 'missing' in name_lower:
            return 'missing_logic'
        else:
            # Try to detect from code
            if re.search(r'assign\s+\w+\s*=\s*1\'b[01];', code):
                return 'stuck_signal'
            return None

def validate_test_suite(suite_module_name: str):
    """Validate an entire test suite"""
    
    log("="*80, "TEST")
    log(f"VALIDATING TEST SUITE: {suite_module_name}", "TEST")
    log("="*80, "TEST")
    
    # Import the test suite
    try:
        if suite_module_name == "extended_test_suite":
            from integration.extended_test_suite import get_extended_test_circuits
            circuits = get_extended_test_circuits()
        elif suite_module_name == "complex_test_suite":
            from integration.complex_test_suite import get_complex_test_circuits
            circuits = get_complex_test_circuits()
        else:
            from integration.comprehensive_vae_test_suite import get_test_circuits
            circuits = get_test_circuits()
    except Exception as e:
        log(f"Failed to import test suite: {e}", "ERROR")
        return
    
    validator = CircuitValidator()
    all_results = []
    
    total_circuits = sum(len(tests) for tests in circuits.values())
    log(f"Total circuits to validate: {total_circuits}\n", "INFO")
    
    circuit_num = 0
    for category, tests in circuits.items():
        log(f"\n{'='*80}", "TEST")
        log(f"CATEGORY: {category}", "TEST")
        log(f"{'='*80}", "TEST")
        
        for name, code, is_normal in tests:
            circuit_num += 1
            log(f"\n[{circuit_num}/{total_circuits}] Validating: {name}", "INFO")
            
            result = validator.validate_circuit(name, code, is_normal, category)
            all_results.append(result)
            
            # Print result
            if result.passed:
                log(f"  ✅ PASSED", "SUCCESS")
            else:
                log(f"  ❌ FAILED - {len(result.issues)} issues", "ERROR")
                for issue in result.issues:
                    log(f"     • {issue}", "ERROR")
            
            if result.warnings:
                for warning in result.warnings:
                    log(f"     ⚠️  {warning}", "WARNING")
    
    # ========================================
    # SUMMARY REPORT
    # ========================================
    
    log("\n" + "="*80, "SUCCESS")
    log("VALIDATION SUMMARY", "TEST")
    log("="*80, "SUCCESS")
    
    passed = sum(1 for r in all_results if r.passed)
    failed = len(all_results) - passed
    
    log(f"\n📊 OVERALL RESULTS:", "INFO")
    log(f"  Total Circuits: {len(all_results)}", "INFO")
    log(f"  Passed: {passed} ({passed/len(all_results)*100:.1f}%)", "SUCCESS")
    log(f"  Failed: {failed} ({failed/len(all_results)*100:.1f}%)", "ERROR" if failed > 0 else "INFO")
    
    # Compilation success rate
    compiled = sum(1 for r in all_results if r.compiles)
    simulated = sum(1 for r in all_results if r.simulation_runs)
    waveforms = sum(1 for r in all_results if r.generates_waveform)
    
    log(f"\n🔧 COMPILATION & SIMULATION:", "INFO")
    log(f"  Compiles: {compiled}/{len(all_results)} ({compiled/len(all_results)*100:.1f}%)", "INFO")
    log(f"  Runs: {simulated}/{len(all_results)} ({simulated/len(all_results)*100:.1f}%)", "INFO")
    log(f"  Generates Waveforms: {waveforms}/{len(all_results)} ({waveforms/len(all_results)*100:.1f}%)", "INFO")
    
    # Structure compliance
    has_all_required = sum(1 for r in all_results if r.has_timescale and r.has_testbench and 
                          r.has_dumpfile and r.has_dumpvars and r.has_finish)
    
    log(f"\n📋 STRUCTURE COMPLIANCE:", "INFO")
    log(f"  Has All Required Elements: {has_all_required}/{len(all_results)} ({has_all_required/len(all_results)*100:.1f}%)", "INFO")
    
    # Sequential circuit quality
    sequential = [r for r in all_results if r.is_sequential]
    combinational = [r for r in all_results if not r.is_sequential]
    
    log(f"\n🔄 CIRCUIT TYPE BREAKDOWN:", "INFO")
    log(f"  Combinational Circuits: {len(combinational)}", "INFO")
    log(f"  Sequential Circuits: {len(sequential)}", "INFO")
    
    if sequential:
        seq_with_clock = sum(1 for r in sequential if r.has_clock)
        seq_clock_dumped = sum(1 for r in sequential if r.clock_in_dumpvars)
        
        log(f"\n⏰ SEQUENTIAL CIRCUIT QUALITY:", "INFO")
        log(f"  Has Clock Signal: {seq_with_clock}/{len(sequential)}", "INFO")
        log(f"  Clock in Waveform: {seq_clock_dumped}/{len(sequential)}", "INFO")
        
        if seq_clock_dumped < len(sequential):
            log(f"  ⚠️  WARNING: {len(sequential) - seq_clock_dumped} sequential circuits don't dump clock!", "WARNING")
    
    # Test vector quality
    avg_vectors = sum(r.num_test_vectors for r in all_results) / len(all_results)
    log(f"\n🧪 TEST VECTOR QUALITY:", "INFO")
    log(f"  Average Test Vectors: {avg_vectors:.1f}", "INFO")
    log(f"  Circuits with <4 vectors: {sum(1 for r in all_results if r.num_test_vectors < 4)}", "WARNING" if any(r.num_test_vectors < 4 for r in all_results) else "INFO")
    
    # Category breakdown
    log(f"\n📁 CATEGORY BREAKDOWN:", "INFO")
    categories = {}
    for r in all_results:
        if r.category not in categories:
            categories[r.category] = {'total': 0, 'passed': 0, 'failed': 0}
        categories[r.category]['total'] += 1
        if r.passed:
            categories[r.category]['passed'] += 1
        else:
            categories[r.category]['failed'] += 1
    
    for cat, stats in sorted(categories.items()):
        pass_rate = stats['passed'] / stats['total'] * 100
        status = "SUCCESS" if pass_rate == 100 else "WARNING" if pass_rate >= 80 else "ERROR"
        log(f"  {cat:30s}: {stats['passed']:2d}/{stats['total']:2d} passed ({pass_rate:5.1f}%)", status)
    
    # List failed circuits
    failed_circuits = [r for r in all_results if not r.passed]
    if failed_circuits:
        log(f"\n❌ FAILED CIRCUITS ({len(failed_circuits)}):", "ERROR")
        for r in failed_circuits:
            log(f"  • {r.circuit_name}", "ERROR")
            for issue in r.issues[:2]:  # Show first 2 issues
                log(f"    - {issue}", "ERROR")
    
    log("\n" + "="*80, "SUCCESS")
    if failed == 0:
        log("🎉 ALL CIRCUITS VALIDATED SUCCESSFULLY!", "SUCCESS")
    elif failed < len(all_results) * 0.1:
        log(f"✅ VALIDATION MOSTLY SUCCESSFUL ({passed}/{len(all_results)} passed)", "SUCCESS")
    else:
        log(f"⚠️  VALIDATION NEEDS ATTENTION ({failed} circuits failed)", "WARNING")
    log("="*80, "SUCCESS")
    
    return all_results

if __name__ == "__main__":
    import sys
    
    # Allow specifying which suite to validate
    if len(sys.argv) > 1:
        suite_name = sys.argv[1]
    else:
        suite_name = "extended_test_suite"  # Default to new suite
    
    validate_test_suite(suite_name)