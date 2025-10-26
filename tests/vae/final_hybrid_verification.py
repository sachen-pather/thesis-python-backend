"""
Final Improved Hybrid Verifier - Addressing Core Issues
Key fixes:
1. Robust output detection using multiple strategies
2. Enhanced logic verification for 2-input gates
3. Better bug pattern detection
4. Smarter decision logic
"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from io import StringIO
import re
from use_vae import load_vae_model, csv_to_features
from services.simulation_service import SimulationService


class FinalHybridVerifier:
    """Final improved verifier with robust I/O detection and logic checking"""
    
    def __init__(self):
        self.model, self.threshold, self.device = load_vae_model()
        self.sim = SimulationService()
        
        self.adaptive_thresholds = {
            'simple_logic': 0.10,
            'complex_logic': 0.12,
            'sequential': 0.16,
            'arithmetic': 0.14,
            'unknown': 0.11
        }
        
    def verify_circuit(self, verilog_code):
        """Main verification with all improvements"""
        try:
            # Run simulation
            success, stdout, _, error = self.sim.simulate_verilog(verilog_code)
            
            if not success:
                return True, 1.0, f"ANOMALOUS: Compilation/simulation failed - {error}"
            
            csv_data = self.parse_monitor_to_csv(stdout)
            if not csv_data:
                return True, 1.0, "ANOMALOUS: No waveform data generated"
            
            df = pd.read_csv(StringIO(csv_data))
            
            # Circuit analysis
            circuit_category = self.detect_circuit_category(df, verilog_code)
            adaptive_threshold = self.adaptive_thresholds.get(circuit_category, self.threshold)
            
            # VAE check
            vae_anomalous, vae_error, vae_message = self.vae_check_adaptive(
                csv_data, adaptive_threshold
            )
            
            # Enhanced rule check
            rule_anomalous, rule_confidence, rule_message = self.comprehensive_rule_check(
                df, verilog_code, circuit_category
            )
            
            # Decision logic with better weighting
            if rule_anomalous and rule_confidence >= 0.7:
                # High confidence detection
                return True, rule_confidence, f"ANOMALOUS - {rule_message}"
            
            elif rule_anomalous and rule_confidence >= 0.5:
                # Medium confidence + VAE confirmation
                if vae_anomalous:
                    return True, 0.8, f"ANOMALOUS - {rule_message} + pattern anomaly"
                else:
                    return True, rule_confidence, f"ANOMALOUS - {rule_message}"
            
            elif not rule_anomalous and rule_confidence >= 0.6:
                # High confidence normal
                return False, 1.0 - rule_confidence, f"NORMAL - {rule_message}"
            
            elif vae_anomalous and vae_error > adaptive_threshold * 1.3:
                # Very strong VAE signal (30% above threshold)
                return True, 0.7, f"ANOMALOUS - Strong pattern anomaly ({vae_error:.3f})"
            
            else:
                # Default to normal
                return False, 0.3, f"NORMAL - {rule_message}, VAE: {vae_error:.3f}"
            
        except Exception as e:
            return True, 1.0, f"ANOMALOUS: Error - {str(e)}"
    
    def comprehensive_rule_check(self, df, verilog_code, circuit_category):
        """Comprehensive rule-based checking with multiple strategies"""
        
        # Strategy 1: Multi-method I/O detection
        inputs, outputs = self.robust_io_detection(df, verilog_code)
        
        print(f"DEBUG: I/O Detection - inputs: {inputs}, outputs: {outputs}")
        print(f"DEBUG: Circuit category: {circuit_category}, len(inputs)={len(inputs)}, len(outputs)={len(outputs)}")
        
        # Strategy 2: Check for obvious bugs (high priority)
        bug_found, bug_conf, bug_msg = self.detect_obvious_bugs(df, inputs, outputs, verilog_code)
        if bug_found and bug_conf >= 0.7:
            return True, bug_conf, bug_msg
        
        # Strategy 3: For simple 2-input logic, try truth table verification
        if len(inputs) == 2 and len(outputs) >= 1 and circuit_category in ['simple_logic', 'unknown']:
            gate_anomaly, gate_conf, gate_msg = self.verify_2input_logic(df, inputs, outputs, verilog_code)
            if gate_anomaly and gate_conf >= 0.6:
                return True, gate_conf, gate_msg
        
        # Strategy 4: Category-specific checks
        if circuit_category == 'sequential':
            return self.verify_sequential_enhanced(df, verilog_code, inputs, outputs)
        elif circuit_category == 'arithmetic':
            return self.verify_arithmetic_enhanced(df, inputs, outputs, verilog_code)
        
        # Strategy 5: If bug was found but low confidence, return it anyway
        if bug_found:
            return True, bug_conf, bug_msg
        
        # Default: appears normal
        return False, 0.6, "behavior appears normal"
    
    def robust_io_detection(self, df, verilog_code):
        """Multi-strategy I/O detection"""
        
        signals = df['signal'].unique()
        inputs = []
        outputs = []
        
        # Method 1: Parse Verilog module declaration
        # Extract the module port list
        module_match = re.search(r'module\s+\w+\s*\((.*?)\);', verilog_code, re.DOTALL | re.IGNORECASE)
        
        if module_match:
            port_list = module_match.group(1)
            
            # Find all input declarations in port list
            for match in re.finditer(r'input\s+(?:wire\s+)?(?:\[[^\]]+\]\s+)?(\w+(?:\s*,\s*\w+)*)', port_list, re.IGNORECASE):
                input_names = [name.strip() for name in match.group(1).split(',')]
                for sig in signals:
                    if any(str(sig).lower() == inp.lower() for inp in input_names):
                        if sig not in inputs:
                            inputs.append(sig)
            
            # Find all output declarations in port list
            for match in re.finditer(r'output\s+(?:wire|reg)?\s+(?:\[[^\]]+\]\s+)?(\w+(?:\s*,\s*\w+)*)', port_list, re.IGNORECASE):
                output_names = [name.strip() for name in match.group(1).split(',')]
                for sig in signals:
                    if any(str(sig).lower() == out.lower() for out in output_names):
                        if sig not in outputs:
                            outputs.append(sig)
        
        # Method 1b: Also check for input/output declarations in module body
        input_matches = re.findall(r'input\s+(?:wire\s+)?(?:\[[^\]]+\]\s+)?(\w+)', verilog_code, re.IGNORECASE)
        output_matches = re.findall(r'output\s+(?:wire|reg)?\s+(?:\[[^\]]+\]\s+)?(\w+)', verilog_code, re.IGNORECASE)
        
        for sig in signals:
            sig_str = str(sig).lower()
            if any(inp.lower() == sig_str for inp in input_matches):
                if sig not in inputs:
                    inputs.append(sig)
            elif any(out.lower() == sig_str for out in output_matches):
                if sig not in outputs:
                    outputs.append(sig)
        # Method 2: If parsing failed or incomplete, use heuristics
        # Make sure we don't miss common input/output patterns
        for sig in signals:
            sig_str = str(sig).lower()
            
            # Skip control signals
            if any(ctrl in sig_str for ctrl in ['clk', 'clock', 'rst', 'reset']):
                continue
            
            # Output patterns (check first as they're more specific)
            output_patterns = ['out', 'result', 'sum', 'carry', 'diff', 'borrow', 
                             'equal', 'greater', 'less', 'valid', 'enc']
            is_output = any(pattern in sig_str for pattern in output_patterns)
            
            # Single letter outputs
            if sig_str in ['y', 'z', 'q', 'o'] and sig not in inputs:
                is_output = True
            
            if is_output and sig not in outputs:
                outputs.append(sig)
            
            # Input patterns - be inclusive
            elif sig not in outputs and sig not in inputs:
                # Common input names
                input_patterns = ['in', 'data', 'sel', 'bit', 'num', 'req', 
                                'trigger', 'up', 'minuend', 'subtrahend', 'cin', 'carry_in', 
                                'borrow_in']
                
                # Single letter inputs
                if sig_str in ['a', 'b', 'c', 'd', 'x', 'y', 's', 's1', 's2', 
                              'i0', 'i1', 'i2', 'i3', 'p', 'q', 'r']:
                    inputs.append(sig)
                elif any(pattern in sig_str for pattern in input_patterns):
                    inputs.append(sig)
        
        # Method 3: Last resort - analyze signal behavior
        if not outputs:
            for sig in signals:
                if 'clk' in str(sig).lower() or 'rst' in str(sig).lower():
                    continue
                
                if sig not in inputs and sig not in outputs:
                    # Signals written in module are typically outputs
                    # Check if signal appears on LHS of assignments
                    sig_pattern = r'(?:<=|=)\s*' + re.escape(str(sig))
                    if re.search(sig_pattern, verilog_code):
                        outputs.append(sig)
                    else:
                        inputs.append(sig)
        
        return inputs, outputs
    
    def detect_obvious_bugs(self, df, inputs, outputs, verilog_code):
        """Detect obvious bugs with high confidence"""
        
        # BUG 1: Stuck-at faults (highest priority)
        if outputs:
            for output in outputs:
                output_data = df[df['signal'] == output]['value']
                unique_values = len(set(str(v) for v in output_data))
                
                if unique_values == 1 and len(output_data) >= 3:
                    # Check if ANY input changes
                    input_changes = False
                    if inputs:
                        for inp in inputs:
                            input_data = df[df['signal'] == inp]['value']
                            if len(set(str(v) for v in input_data)) > 1:
                                input_changes = True
                                break
                    
                    if input_changes or not inputs:
                        const_val = str(output_data.iloc[0])
                        return True, 0.9, f"output '{output}' stuck at {const_val}"
        
        # BUG 2: Unresponsive outputs
        if inputs and outputs:
            input_activity = sum(1 for inp in inputs 
                                if len(set(str(v) for v in df[df['signal'] == inp]['value'])) > 1)
            output_activity = sum(1 for out in outputs 
                                 if len(set(str(v) for v in df[df['signal'] == out]['value'])) > 1)
            
            if input_activity > 0 and output_activity == 0:
                return True, 0.8, "outputs don't respond to inputs"
        
        # BUG 3: Code pattern analysis
        code_lower = verilog_code.lower()
        
        # Hardcoded outputs in always blocks
        if 'always' in code_lower:
            always_blocks = re.findall(r'always\s*@[^;]+begin(.+?)end', verilog_code, re.DOTALL | re.IGNORECASE)
            for block in always_blocks:
                block_lower = block.lower()
                # Single assignment to constant
                if '<=' in block_lower and block_lower.count('<=') == 1:
                    if "1'b0" in block_lower or "1'b1" in block_lower:
                        return True, 0.75, "always block with constant assignment"
        
        # Suspicious assign statements
        if 'assign' in code_lower:
            assigns = re.findall(r'assign\s+\w+\s*=\s*([^;]+);', verilog_code, re.IGNORECASE)
            for rhs in assigns:
                rhs_lower = rhs.lower().strip()
                if rhs_lower in ["1'b0", "1'b1", "0", "1"]:
                    return True, 0.7, "assign to constant value"
        
        return False, 0.0, "no obvious bugs"
    
    def verify_2input_logic(self, df, inputs, outputs, verilog_code):
        """Verify 2-input logic gates with truth table checking"""
        
        if len(inputs) != 2 or len(outputs) == 0:
            return False, 0.0, ""
        
        output = outputs[0]
        
        # Build truth table from simulation
        truth_table = {}
        for timestamp in df['timestamp'].unique():
            row = df[df['timestamp'] == timestamp]
            
            inp_vals = []
            out_val = None
            
            for inp in inputs:
                inp_data = row[row['signal'] == inp]
                if not inp_data.empty:
                    val = str(inp_data.iloc[0]['value'])
                    inp_vals.append(1 if val == '1' else 0)
            
            out_data = row[row['signal'] == output]
            if not out_data.empty:
                val = str(out_data.iloc[0]['value'])
                out_val = 1 if val == '1' else 0
            
            if len(inp_vals) == 2 and out_val is not None:
                key = tuple(inp_vals)
                truth_table[key] = out_val
        
        if len(truth_table) < 3:  # Need at least 3 combinations
            return False, 0.0, ""
        
        # Analyze what gate this implements
        gate_type = self.identify_gate_from_truth_table(truth_table)
        
        # Check Verilog code to see what gate was INTENDED
        code_lower = verilog_code.lower()
        
        intended_gate = None
        if 'nand' in code_lower:
            intended_gate = 'nand'
        elif 'nor' in code_lower:
            intended_gate = 'nor'
        elif '~(' in code_lower or '!(' in code_lower:
            if '&' in code_lower:
                intended_gate = 'nand'
            elif '|' in code_lower:
                intended_gate = 'nor'
        elif '&' in code_lower:
            intended_gate = 'and'
        elif '|' in code_lower:
            intended_gate = 'or'
        elif '^' in code_lower:
            intended_gate = 'xor'
        
        # Compare intended vs actual
        if intended_gate and gate_type and intended_gate != gate_type:
            return True, 0.8, f"implements {gate_type} instead of {intended_gate}"
        
        return False, 0.0, ""
    
    def identify_gate_from_truth_table(self, tt):
        """Identify gate type from truth table"""
        
        # Standard truth tables
        gates = {
            'and': {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1},
            'or': {(0,0): 0, (0,1): 1, (1,0): 1, (1,1): 1},
            'xor': {(0,0): 0, (0,1): 1, (1,0): 1, (1,1): 0},
            'nand': {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 0},
            'nor': {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0},
        }
        
        for gate_name, expected_tt in gates.items():
            match_count = sum(1 for k in tt if k in expected_tt and tt[k] == expected_tt[k])
            if match_count == len(tt) and len(tt) >= 3:
                return gate_name
        
        return 'unknown'
    
    def verify_sequential_enhanced(self, df, verilog_code, inputs, outputs):
        """Enhanced sequential verification"""
        
        has_clock = self.detect_clock_signal(df)
        if not has_clock:
            return False, 0.4, "sequential without detectable clock"
        
        # Check for state changes
        state_signals = [s for s in df['signal'].unique() 
                        if 'clk' not in str(s).lower() and 'rst' not in str(s).lower()]
        
        if not state_signals:
            return False, 0.4, "no state signals found"
        
        # Check if any state signal changes
        any_changes = False
        for sig in state_signals:
            sig_data = df[df['signal'] == sig]['value']
            if len(set(str(v) for v in sig_data)) > 1:
                any_changes = True
                break
        
        if not any_changes:
            # Check if there are inputs that should cause changes
            if inputs:
                input_activity = sum(1 for inp in inputs if len(set(str(v) for v in df[df['signal'] == inp]['value'])) > 1)
                if input_activity > 0:
                    return True, 0.7, "sequential circuit with no state transitions"
        
        return False, 0.5, "sequential behavior appears normal"
    
    def verify_arithmetic_enhanced(self, df, inputs, outputs, verilog_code):
        """Enhanced arithmetic verification"""
        
        if not outputs:
            return False, 0.3, "arithmetic without detectable outputs"
        
        # Check output responsiveness
        output_changes = sum(1 for out in outputs 
                            if len(set(str(v) for v in df[df['signal'] == out]['value'])) > 1)
        
        if output_changes == 0:
            input_changes = 0
            if inputs:
                input_changes = sum(1 for inp in inputs 
                                   if len(set(str(v) for v in df[df['signal'] == inp]['value'])) > 1)
            
            if input_changes > 0:
                return True, 0.7, "arithmetic outputs unresponsive"
        
        return False, 0.6, "arithmetic behavior appears normal"
    
    def detect_circuit_category(self, df, verilog_code):
        """Improved circuit category detection"""
        signals = df['signal'].unique()
        verilog_lower = verilog_code.lower()
        
        has_clock = self.detect_clock_signal(df)
        has_multi_bit = any('[' in str(s) for s in signals)
        num_signals = len([s for s in signals if 'clk' not in str(s).lower() and 'rst' not in str(s).lower()])
        
        arithmetic_keywords = ['carry', 'sum', 'borrow', 'diff', 'comparator', 'adder', 'subtractor']
        has_arithmetic = any(kw in verilog_lower for kw in arithmetic_keywords)
        
        # Priority: Sequential must have BOTH clock AND reg/always
        if has_clock and self.has_sequential_pattern(verilog_code):
            return 'sequential'
        # Arithmetic: has arithmetic keywords or multiple outputs
        elif has_arithmetic:
            return 'arithmetic'
        # Complex logic: many signals
        elif num_signals > 4:
            return 'complex_logic'
        # Simple logic: default for combinational circuits
        else:
            return 'simple_logic'
    
    def has_sequential_pattern(self, verilog_code):
        """Check if code has sequential patterns - must have state storage"""
        # Extract only the DUT module, not testbench
        dut_module = self.extract_dut_module(verilog_code)
        if not dut_module:
            dut_module = verilog_code
        
        code_lower = dut_module.lower()
        
        # Must have BOTH always block with edge sensitivity AND output register
        has_posedge_always = '@(posedge' in code_lower and 'always' in code_lower
        has_output_reg = 'output reg' in code_lower
        
        # Simple assign statements are NOT sequential
        return has_posedge_always and has_output_reg
    
    def extract_dut_module(self, verilog_code):
        """Extract the first non-testbench module (the DUT)"""
        # Find all modules
        modules = re.findall(r'module\s+(\w+).*?endmodule', verilog_code, re.DOTALL | re.IGNORECASE)
        
        if not modules:
            return None
        
        # Find first module that's not called 'testbench' or 'tb'
        for match in re.finditer(r'module\s+(\w+)(.*?)endmodule', verilog_code, re.DOTALL | re.IGNORECASE):
            module_name = match.group(1).lower()
            if 'test' not in module_name and 'tb' != module_name:
                return match.group(0)
        
        return None
    
    def detect_clock_signal(self, df):
        """Detect clock signals"""
        for signal in df['signal'].unique():
            sig_str = str(signal).lower()
            if 'clk' in sig_str or 'clock' in sig_str:
                return True
            
            # Check for regular toggling
            signal_data = df[df['signal'] == signal].sort_values('timestamp')
            if len(signal_data) < 4:
                continue
            
            values = [int(v) if str(v) in ['0', '1'] else 0 for v in signal_data['value']]
            if len(values) >= 4:
                transitions = sum(1 for i in range(1, len(values)) if values[i] != values[i-1])
                if transitions >= len(values) * 0.4:
                    return True
        
        return False
    
    def vae_check_adaptive(self, csv_data, threshold):
        """VAE check with adaptive threshold"""
        try:
            features = csv_to_features(csv_data)
            if features is None:
                return True, 1.0, "feature extraction failed"
            
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            with torch.no_grad():
                x_recon, _, _ = self.model(x)
                error = F.mse_loss(x, x_recon).item()
            
            is_anomalous = error > threshold
            message = f"error {error:.3f}"
            
            return is_anomalous, error, message
        except Exception as e:
            return True, 1.0, f"VAE error: {str(e)}"
    
    def parse_monitor_to_csv(self, stdout):
        """Parse $monitor output"""
        lines = stdout.strip().split('\n')
        csv_lines = ['timestamp,signal,value']
        
        for line in lines:
            if not line.startswith('Time='):
                continue
            parts = line.split()
            timestamp = None
            for part in parts:
                if '=' in part:
                    key, val = part.split('=', 1)
                    if key == 'Time':
                        timestamp = val
                    elif timestamp:
                        csv_lines.append(f"{timestamp},{key},{val}")
        
        return '\n'.join(csv_lines) if len(csv_lines) > 1 else None


def verify_circuit_final(verilog_code):
    """Main function for final improved verification"""
    verifier = FinalHybridVerifier()
    return verifier.verify_circuit(verilog_code)