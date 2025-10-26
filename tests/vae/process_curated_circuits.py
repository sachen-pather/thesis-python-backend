# tests/vae/process_curated_circuits.py
"""
Complete pipeline to process curated Verilog circuits into VAE training data
"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from services.simulation_service import SimulationService
import pandas as pd
from io import StringIO
import pickle
from datetime import datetime
import glob


def parse_monitor_to_csv(stdout):
    """Parse $monitor output to CSV format"""
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
    return '\n'.join(csv_lines)


def verify_waveform(csv_data):
    """Basic verification that waveform has changing signals"""
    try:
        df = pd.read_csv(StringIO(csv_data))
        
        if len(df) < 5:
            return False, "Too few data points"
        
        # Check that at least one signal changes
        for signal in df['signal'].unique():
            signal_data = df[df['signal'] == signal]['value']
            unique_vals = len(set(str(v) for v in signal_data))
            if unique_vals > 1:
                return True, f"Valid: {signal} changes states"
        
        return False, "No signals change"
        
    except Exception as e:
        return False, f"Verification error: {str(e)}"


class CuratedCircuitProcessor:
    """Process curated Verilog circuits into training data"""
    
    def __init__(self, circuits_dir="curated_circuits"):
        self.circuits_dir = circuits_dir
        self.sim = SimulationService()
        self.processed_circuits = []
        self.failed_circuits = []
        
        # Create circuits directory if it doesn't exist
        os.makedirs(circuits_dir, exist_ok=True)
        
    def process_all_circuits(self):
        """Process all .v files in the circuits directory"""
        verilog_files = glob.glob(f"{self.circuits_dir}/*.v")
        
        if not verilog_files:
            print(f"No .v files found in {self.circuits_dir}/")
            print("Please add your 18 curated circuit files to this directory")
            return
        
        print(f"Found {len(verilog_files)} Verilog files")
        print("Processing circuits...")
        print("="*60)
        
        for verilog_file in verilog_files:
            circuit_name = os.path.basename(verilog_file).replace('.v', '')
            print(f"\nProcessing: {circuit_name}")
            
            # Read Verilog file
            try:
                with open(verilog_file, 'r') as f:
                    verilog_code = f.read()
            except Exception as e:
                print(f"  ❌ Failed to read file: {e}")
                self.failed_circuits.append({'name': circuit_name, 'error': f"File read error: {e}"})
                continue
            
            # Simulate circuit
            success, stdout, csv_data, error = self.sim.simulate_verilog(verilog_code)
            
            if not success:
                print(f"  ❌ Simulation failed: {error}")
                self.failed_circuits.append({'name': circuit_name, 'error': f"Simulation: {error}"})
                continue
            
            # Convert monitor output to CSV
            if not csv_data and stdout:
                csv_data = parse_monitor_to_csv(stdout)
            
            if not csv_data or len(csv_data) < 50:
                print(f"  ❌ No valid CSV data generated")
                self.failed_circuits.append({'name': circuit_name, 'error': "No CSV data"})
                continue
            
            # Verify waveform
            is_valid, message = verify_waveform(csv_data)
            
            if not is_valid:
                print(f"  ❌ Verification failed: {message}")
                self.failed_circuits.append({'name': circuit_name, 'error': f"Verification: {message}"})
                continue
            
            # Success - add to training data
            circuit_data = {
                'name': circuit_name,
                'type': self._infer_circuit_type(circuit_name),
                'verilog': verilog_code,
                'csv': csv_data,
                'verification_message': message,
                'timestamp': datetime.now().isoformat(),
                'source': 'curated'
            }
            
            self.processed_circuits.append(circuit_data)
            print(f"  ✅ Success: {message}")
        
        self._print_summary()
    
    def _infer_circuit_type(self, circuit_name):
        """Infer circuit type from filename"""
        name_lower = circuit_name.lower()
        
        if 'and' in name_lower:
            return 'curated_and_gate'
        elif 'or' in name_lower:
            return 'curated_or_gate'
        elif 'xor' in name_lower:
            return 'curated_xor_gate'
        elif 'mux' in name_lower:
            return 'curated_multiplexer'
        elif 'demux' in name_lower:
            return 'curated_demultiplexer'
        elif 'decoder' in name_lower:
            return 'curated_decoder'
        elif 'd_flip' in name_lower or 'd_ff' in name_lower:
            return 'curated_d_flip_flop'
        elif 'jk' in name_lower:
            return 'curated_jk_flip_flop'
        elif 't_flip' in name_lower or 'toggle' in name_lower:
            return 'curated_toggle_flip_flop'
        elif 'sr' in name_lower:
            return 'curated_sr_flip_flop'
        elif 'latch' in name_lower:
            return 'curated_latch'
        elif 'counter' in name_lower:
            return 'curated_counter'
        elif 'shift' in name_lower:
            return 'curated_shift_register'
        elif 'register' in name_lower:
            return 'curated_register'
        elif 'adder' in name_lower:
            return 'curated_adder'
        elif 'pipeline' in name_lower:
            return 'curated_pipeline'
        elif 'state' in name_lower or 'fsm' in name_lower:
            return 'curated_state_machine'
        else:
            return 'curated_unknown'
    
    def _print_summary(self):
        """Print processing summary"""
        print("\n" + "="*60)
        print("CURATED CIRCUIT PROCESSING SUMMARY")
        print("="*60)
        print(f"✅ Successfully processed: {len(self.processed_circuits)}")
        print(f"❌ Failed circuits: {len(self.failed_circuits)}")
        print(f"📊 Success rate: {len(self.processed_circuits)/(len(self.processed_circuits) + len(self.failed_circuits))*100:.1f}%")
        
        if self.processed_circuits:
            from collections import Counter
            types = Counter([c['type'] for c in self.processed_circuits])
            print("\nCircuit Distribution:")
            for circuit_type, count in sorted(types.items()):
                print(f"  {circuit_type}: {count}")
        
        if self.failed_circuits:
            print(f"\nFailed Circuits:")
            for failed in self.failed_circuits:
                print(f"  ❌ {failed['name']}: {failed['error']}")
        
        print("="*60)
    
    def save_training_data(self, filename='curated_training_data.pkl'):
        """Save processed circuits as training data"""
        if not self.processed_circuits:
            print("No circuits to save!")
            return
        
        with open(filename, 'wb') as f:
            pickle.dump(self.processed_circuits, f)
        
        print(f"\n✅ Saved {len(self.processed_circuits)} circuits to {filename}")
        return filename
    
    def combine_with_existing_data(self, existing_file='enhanced_training_data.pkl'):
        """Combine curated data with existing training data"""
        existing_data = []
        
        if existing_file:
            try:
                with open(existing_file, 'rb') as f:
                    existing_data = pickle.load(f)
                print(f"📂 Loaded {len(existing_data)} existing circuits from {existing_file}")
            except FileNotFoundError:
                print(f"📂 No existing file {existing_file} found")
        
        combined_data = existing_data + self.processed_circuits
        return combined_data
    
    def export_csv_files(self, output_dir="circuit_csvs"):
        """Export individual CSV files for inspection"""
        os.makedirs(output_dir, exist_ok=True)
        
        for circuit in self.processed_circuits:
            csv_filename = f"{output_dir}/{circuit['name']}.csv"
            with open(csv_filename, 'w') as f:
                f.write(circuit['csv'])
        
        print(f"📁 Exported {len(self.processed_circuits)} CSV files to {output_dir}/")


def setup_directory_structure():
    """Create the required directory structure and instructions"""
    
    print("SETTING UP CURATED CIRCUIT PROCESSING")
    print("="*50)
    
    # Create directories
    os.makedirs("curated_circuits", exist_ok=True)
    os.makedirs("circuit_csvs", exist_ok=True)
    
    print("✅ Created directory structure:")
    print("   tests/vae/curated_circuits/  <- Put your 18 .v files here")
    print("   tests/vae/circuit_csvs/      <- CSV outputs will go here")
    
    # Create example file structure
    example_files = [
        "and_gate.v",
        "or_gate.v", 
        "xor_gate.v",
        "mux_4x1.v",
        "demux_1x4.v",
        "decoder_3x8.v",
        "d_flip_flop.v",
        "jk_flip_flop.v",
        "sr_flip_flop.v",
        "t_flip_flop.v",
        "sr_latch.v",
        "counter_4bit.v",
        "shift_register.v",
        "register_8bit.v",
        "full_adder.v",
        "half_adder.v",
        "pipeline.v",
        "state_machine.v"
    ]
    
    print(f"\n📋 Expected files (copy your curated circuits here):")
    for i, filename in enumerate(example_files, 1):
        exists = "✅" if os.path.exists(f"curated_circuits/{filename}") else "⭕"
        print(f"   {exists} {i:2d}. curated_circuits/{filename}")
    
    print(f"\n📝 Instructions:")
    print("1. Copy your 18 curated .v files to curated_circuits/")
    print("2. Run: python process_curated_circuits.py")
    print("3. Training data will be saved as curated_training_data.pkl")
    print("4. Update vae_mvp.py to load the new training data")
    

def main():
    """Main processing pipeline"""
    
    # Setup directory structure first
    setup_directory_structure()
    
    # Check if we have circuit files
    verilog_files = glob.glob("curated_circuits/*.v")
    if not verilog_files:
        print(f"\n⚠️  No .v files found in curated_circuits/")
        print("Please add your 18 curated circuit files first, then re-run this script.")
        return
    
    print(f"\n🚀 Processing {len(verilog_files)} curated circuits...")
    
    # Process circuits
    processor = CuratedCircuitProcessor()
    processor.process_all_circuits()
    
    if processor.processed_circuits:
        # Save training data
        training_file = processor.save_training_data()
        
        # Export CSV files for inspection
        processor.export_csv_files()
        
        # Instructions for next steps
        print(f"\n📋 NEXT STEPS:")
        print(f"1. Inspect CSV files in circuit_csvs/ directory")
        print(f"2. Update vae_mvp.py to load '{training_file}':")
        print(f"   Change: with open('enhanced_training_data.pkl', 'rb') as f:")
        print(f"   To:     with open('{training_file}', 'rb') as f:")
        print(f"3. Retrain VAE: python vae_mvp.py")
        print(f"4. Evaluate: python evaluate_vae_performance.py")
    else:
        print("\n❌ No circuits were successfully processed!")
        print("Check the error messages above and fix the circuit files.")


if __name__ == "__main__":
    main()