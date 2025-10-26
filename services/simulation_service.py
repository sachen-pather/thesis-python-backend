# services/simulation_service.py
import tempfile
import subprocess
import time
import os
import pandas as pd
from datetime import datetime
from io import StringIO

class SimulationService:
    def __init__(self):
        self.compilation_log = ""
        self.last_simulation_time = 0.0
    
    def log_message(self, message, level="info"):
        """Add timestamped log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.compilation_log += f"[{timestamp}] {level.upper()}: {message}\n"

    def simulate_verilog(self, verilog_code, is_systemverilog=False):
        """Enhanced simulation with SystemVerilog support"""
        self.log_message("Starting simulation process...")
        
        # Clean and validate the code
        verilog_code = verilog_code.strip()
        
        # Ensure timescale is at the top
        if "`timescale" not in verilog_code:
            verilog_code = "`timescale 1ns/1ps\n\n" + verilog_code
        else:
            lines = verilog_code.split('\n')
            timescale_idx = next((i for i, line in enumerate(lines) if "`timescale" in line), -1)
            if timescale_idx > 0:
                timescale = lines.pop(timescale_idx)
                verilog_code = timescale + "\n\n" + "\n".join(lines)
        
        # Auto-detect SystemVerilog
        if not is_systemverilog:
            sv_keywords = ['logic', 'always_comb', 'always_ff', 'always_latch', 'typedef', 'interface']
            is_systemverilog = any(keyword in verilog_code for keyword in sv_keywords)
            if is_systemverilog:
                self.log_message("Detected SystemVerilog constructs, using .sv extension")
        
        # Remove potential encoding issues
        verilog_code = verilog_code.encode('ascii', 'ignore').decode('ascii')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use .sv extension for SystemVerilog, .v for Verilog
            file_ext = ".sv" if is_systemverilog else ".v"
            design_path = os.path.join(tmpdir, f"design{file_ext}")
            
            # Write file
            try:
                with open(design_path, "w", encoding='ascii', newline='\n') as f:
                    f.write(verilog_code)
                self.log_message(f"Code written to {design_path}")
                
            except Exception as e:
                self.log_message(f"File writing failed: {str(e)}", "error")
                return False, f"❌ File Writing Error:\n{str(e)}", "", ""
            
            # Compilation
            self.log_message("Compiling design...")
            
            # For SystemVerilog, add -g2009 or -g2012 flag
            compile_cmd = ["iverilog", "-o", os.path.join(tmpdir, "sim"), design_path]
            if is_systemverilog:
                # Try with SystemVerilog support
                compile_cmd.insert(1, "-g2012")  # SystemVerilog-2012 standard
                self.log_message("Using SystemVerilog-2012 mode")
            
            try:
                result = subprocess.run(
                    compile_cmd, 
                    capture_output=True, 
                    text=True, 
                    cwd=tmpdir,
                    timeout=30
                )
            except subprocess.TimeoutExpired:
                self.log_message("Compilation timeout", "error")
                return False, "❌ Compilation timeout (>30s)", "", ""
            except Exception as e:
                self.log_message(f"Compilation process failed: {str(e)}", "error")
                return False, f"❌ Compilation Process Error:\n{str(e)}", "", ""
            
            if result.returncode != 0:
                error_msg = result.stderr.replace(tmpdir + "/", "").replace(tmpdir, "")
                self.log_message(f"Compilation failed: {error_msg}", "error")
                return False, f"❌ Compilation Error:\n{error_msg}", "", ""
            
            self.log_message("Compilation successful")
            
            # Simulation
            self.log_message("Running simulation...")
            start_time = time.time()
            
            sim_cmd = ["vvp", os.path.join(tmpdir, "sim")]
            
            try:
                result = subprocess.run(
                    sim_cmd, 
                    capture_output=True, 
                    text=True, 
                    cwd=tmpdir,
                    timeout=60
                )
            except subprocess.TimeoutExpired:
                self.log_message("Simulation timeout", "error")
                return False, "❌ Simulation timeout (>60s)", "", ""
            except Exception as e:
                self.log_message(f"Simulation process failed: {str(e)}", "error")
                return False, f"❌ Simulation Process Error:\n{str(e)}", "", ""
                
            sim_time = time.time() - start_time
            
            if result.returncode != 0:
                self.log_message(f"Simulation failed: {result.stderr}", "error")
                return False, f"❌ Simulation Error:\n{result.stderr}", "", ""
            
            self.log_message(f"Simulation completed in {sim_time:.2f}s")
            self.last_simulation_time = sim_time
            
            # Check for VCD file
            import glob
            vcd_files = glob.glob(os.path.join(tmpdir, "*.vcd"))
            
            if not vcd_files:
                self.log_message("No VCD file generated", "warning")
                
                # Try fallback: parse $monitor output
                csv_from_output = self.parse_simulation_output_to_csv(result.stdout)
                if csv_from_output:
                    self.log_message("Using simulation monitor output as waveform data")
                    return True, result.stdout, csv_from_output, ""
                
                return True, result.stdout, "", "⚠️ No waveform data generated"
            
            vcd_path = vcd_files[0]
            vcd_filename = os.path.basename(vcd_path)
            
            # Verify VCD file size
            vcd_size = os.path.getsize(vcd_path)
            self.log_message(f"Found VCD file '{vcd_filename}': {vcd_size} bytes")
            
            if vcd_size == 0:
                self.log_message("VCD file is empty", "warning")
                return True, result.stdout, "", "⚠️ Empty VCD file generated"
            
            # Convert VCD to CSV
            self.log_message("Converting VCD to CSV...")
            try:
                csv_content = self.convert_vcd_to_csv(vcd_path)
                
                if "conversion_failed" in csv_content or "error" in csv_content:
                    self.log_message("VCD conversion failed, trying simulation output parsing", "warning")
                    csv_from_output = self.parse_simulation_output_to_csv(result.stdout)
                    if csv_from_output:
                        self.log_message("Successfully extracted waveform from simulation output")
                        return True, result.stdout, csv_from_output, ""
                
                self.log_message("VCD conversion successful")
                return True, result.stdout, csv_content, ""
            except Exception as e:
                self.log_message(f"VCD conversion failed: {str(e)}", "warning")
                csv_from_output = self.parse_simulation_output_to_csv(result.stdout)
                if csv_from_output:
                    return True, result.stdout, csv_from_output, ""
                else:
                    return True, result.stdout, "", f"⚠️ CSV conversion failed: {str(e)}"

    def parse_simulation_output_to_csv(self, simulation_output):
        """Extract waveform data directly from simulation monitor output"""
        try:
            self.log_message("Attempting to parse simulation monitor output")
            
            csv_lines = ["timestamp,signal,value"]
            lines = simulation_output.split('\n')
            
            for line in lines:
                if line.startswith('Time=') and '=' in line:
                    parts = line.split()
                    timestamp = None
                    signal_values = {}
                    
                    for part in parts:
                        if part.startswith('Time='):
                            timestamp = part.split('=')[1]
                        elif '=' in part:
                            signal, value = part.split('=')
                            signal_values[signal] = value
                    
                    if timestamp and signal_values:
                        for signal, value in signal_values.items():
                            csv_lines.append(f"{timestamp},{signal},{value}")
            
            if len(csv_lines) > 1:
                self.log_message(f"Successfully parsed {len(csv_lines)-1} data points")
                return "\n".join(csv_lines)
            
        except Exception as e:
            self.log_message(f"Simulation output parsing failed: {str(e)}", "warning")
        
        return None

    def convert_vcd_to_csv(self, vcd_path):
        """Simple VCD to CSV converter - no external dependencies needed"""
        try:
            self.log_message("Starting VCD to CSV conversion...")
            
            with open(vcd_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            self.log_message(f"Read {len(lines)} lines from VCD file")
            
            # Step 1: Extract signal definitions from VCD header
            signals = {}  # Maps identifier (like !, ", #) to signal name
            for line in lines:
                if '$var' in line:
                    # Format: $var reg 2 ! state [1:0] $end
                    # or:     $var wire 1 " clk $end
                    parts = line.split()
                    if len(parts) >= 5:
                        identifier = parts[3]  # The symbol: !, ", #, etc.
                        signal_name = parts[4].replace('$end', '').strip()
                        signals[identifier] = signal_name
                        self.log_message(f"  Found signal: {signal_name} -> {identifier}")
            
            if not signals:
                self.log_message("No signals found in VCD file", "error")
                return "timestamp,signal,value\n0,error,No signals found"
            
            self.log_message(f"Parsed {len(signals)} signals successfully")
            
            # Step 2: Extract value changes from VCD body
            csv_lines = ['timestamp,signal,value']
            current_time = '0'
            value_count = 0
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and VCD directives
                if not line or line.startswith('$'):
                    continue
                
                # Time marker: #10, #100, etc.
                if line.startswith('#'):
                    current_time = line[1:]
                    continue
                
                # Multi-bit value: "b1010 !" or "b00 ""
                if line.startswith('b'):
                    parts = line.split()
                    if len(parts) >= 2:
                        value = parts[0][1:]  # Remove 'b' prefix
                        identifier = parts[1]
                        if identifier in signals:
                            csv_lines.append(f"{current_time},{signals[identifier]},{value}")
                            value_count += 1
                
                # Single-bit value: "1!" or "0"" or "x#"
                elif len(line) >= 2 and line[0] in '01xzXZ':
                    value = line[0]
                    identifier = line[1:].strip()
                    if identifier in signals:
                        csv_lines.append(f"{current_time},{signals[identifier]},{value}")
                        value_count += 1
                
                # Real number value: "r1.5 !" (less common)
                elif line.startswith('r'):
                    parts = line.split()
                    if len(parts) >= 2:
                        value = parts[0][1:]  # Remove 'r' prefix
                        identifier = parts[1]
                        if identifier in signals:
                            csv_lines.append(f"{current_time},{signals[identifier]},{value}")
                            value_count += 1
            
            if value_count == 0:
                self.log_message("No value changes found in VCD", "warning")
                return "timestamp,signal,value\n0,error,No value changes found"
            
            result = '\n'.join(csv_lines)
            self.log_message(f"Successfully converted VCD to CSV: {value_count} value changes")
            
            return result
            
        except Exception as e:
            error_msg = f"VCD parsing failed: {str(e)}"
            self.log_message(error_msg, "error")
            return f"timestamp,signal,value\n0,error,{str(e)}"

    def check_iverilog_available(self):
        """Check if iverilog is available"""
        try:
            result = subprocess.run(["iverilog", "-V"], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def get_iverilog_version(self):
        """Get iverilog version"""
        try:
            result = subprocess.run(["iverilog", "-V"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.split('\n')[0]
            return "Unknown"
        except:
            return "Not available"
    
    def get_compilation_log(self):
        return self.compilation_log
    
    def get_last_simulation_time(self):
        return self.last_simulation_time