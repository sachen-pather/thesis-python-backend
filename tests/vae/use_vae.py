# tests/vae/use_vae.py
"""
Simple script to use the trained VAE for verification
FIXED: Now uses the same feature extraction as vae_mvp.py
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
from services.simulation_service import SimulationService


class SimpleVAE(torch.nn.Module):
    def __init__(self, input_dim=128, latent_dim=16):
        super().__init__()
        self.enc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 32), torch.nn.ReLU()
        )
        self.fc_mu = torch.nn.Linear(32, latent_dim)
        self.fc_logvar = torch.nn.Linear(32, latent_dim)
        
        self.dec = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, input_dim), torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        h = self.enc(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -5, 5)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        return self.dec(z), mu, logvar


def csv_to_features(csv_data, feature_dim=128):
    """
    Behavioral feature extraction with proper numerical stability
    MUST MATCH THE TRAINING IMPLEMENTATION IN vae_mvp.py
    """
    try:
        if not csv_data or len(csv_data.strip()) < 10:
            return np.zeros(feature_dim, dtype=np.float32)
            
        df = pd.read_csv(StringIO(csv_data))
        if df.empty:
            return np.zeros(feature_dim, dtype=np.float32)
        
        # Sort by timestamp to ensure proper temporal analysis
        df = df.sort_values('timestamp')
        
        # Initialize feature list
        all_features = []
        
        # 1. BEHAVIORAL PATTERN ANALYSIS 
        signals = sorted(df['signal'].unique())
        
        for signal in signals:
            signal_data = df[df['signal'] == signal].sort_values('timestamp')
            if len(signal_data) < 2:
                continue
                
            values = extract_signal_values(signal_data)
            if len(values) < 2:
                continue
            
            # Safe behavioral pattern features with bounds checking
            all_features.extend([
                # State transition rate (0 to 1)
                min(1.0, np.sum(np.diff(values) != 0) / max(len(values), 1)),
                
                # Value stability (0 to 1)
                min(1.0, np.sum(np.diff(values) == 0) / max(len(values), 1)),
                
                # Monotonic patterns (0 or 1)
                float(np.all(np.diff(values) >= 0)) if len(values) > 1 else 0.0,
                float(np.all(np.diff(values) <= 0)) if len(values) > 1 else 0.0,
                
                # Value range utilization (0 to 1)
                safe_divide(np.max(values) - np.min(values), np.max(values) + 1e-8),
                
                # Duty cycle for binary signals (0 to 1)
                np.mean(values > 0.5) if len(values) > 0 else 0.0,
                
                # Signal variance (bounded)
                min(10.0, np.var(values)) if len(values) > 0 else 0.0,
                
                # Simple periodicity measure (0 to 1)
                safe_periodicity(values),
            ])
        
        # 2. SIMPLE TEMPORAL FEATURES
        # Look for clock-like signals
        clock_signal = find_clock_signal_simple(df, signals)
        
        if clock_signal:
            clock_data = df[df['signal'] == clock_signal]
            clock_values = extract_signal_values(clock_data)
            
            # Clock-related features (bounded)
            all_features.extend([
                # Clock transition rate
                min(1.0, np.sum(np.diff(clock_values) != 0) / max(len(clock_values), 1)),
                
                # Clock duty cycle
                np.mean(clock_values > 0.5) if len(clock_values) > 0 else 0.0,
            ])
        
        # 3. CIRCUIT-TYPE DETECTION (simplified)
        all_features.extend([
            # Has counter-like signal names
            float(any('count' in str(s).lower() for s in signals)),
            
            # Has clock signal
            float(any('clk' in str(s).lower() for s in signals)),
            
            # Has reset signal  
            float(any('rst' in str(s).lower() or 'reset' in str(s).lower() for s in signals)),
            
            # Number of signals (normalized)
            min(1.0, len(signals) / 10.0),
        ])
        
        # 4. RAW SIGNAL SAMPLING (bounded)
        for signal in signals[:3]:  # Limit to first 3 signals
            signal_data = df[df['signal'] == signal].sort_values('timestamp')
            values = extract_signal_values(signal_data)
            
            if len(values) >= 3:
                # Normalize values to [0, 1] range
                if np.max(values) > np.min(values):
                    normalized = (values - np.min(values)) / (np.max(values) - np.min(values))
                else:
                    normalized = np.zeros_like(values)
                
                # Sample key points
                all_features.extend([
                    normalized[0],                    # Initial value
                    normalized[len(normalized)//2],   # Middle value  
                    normalized[-1],                   # Final value
                    np.mean(normalized),              # Average
                ])
        
        # Ensure we have enough features, pad with zeros if needed
        while len(all_features) < feature_dim:
            all_features.append(0.0)
        
        # Convert to numpy array and trim to exact size
        features = np.array(all_features[:feature_dim], dtype=np.float32)
        
        # CRITICAL: Final safety checks
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        features = np.clip(features, -10.0, 10.0)  # Hard bounds
        
        # Optional: Additional normalization
        feature_max = np.max(np.abs(features))
        if feature_max > 1.0:
            features = features / feature_max
        
        return features
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return np.zeros(feature_dim, dtype=np.float32)


def safe_divide(numerator, denominator):
    """Safe division with bounds"""
    if abs(denominator) < 1e-8:
        return 0.0
    result = numerator / denominator
    return np.clip(result, 0.0, 1.0)


def safe_periodicity(values):
    """Simple periodicity measure with bounds"""
    if len(values) < 4:
        return 0.0
    
    try:
        # Simple alternating pattern detection
        alternating = 0
        for i in range(2, min(len(values), 10)):  # Limit to first 10 values
            if values[i] == values[i-2] and values[i] != values[i-1]:
                alternating += 1
        
        return min(1.0, alternating / max(len(values) - 2, 1))
    except:
        return 0.0


def find_clock_signal_simple(df, signals):
    """Simple clock signal detection"""
    for signal in signals:
        if 'clk' in str(signal).lower():
            return signal
    return None


def extract_signal_values(signal_data):
    """Extract numeric values from signal data with error handling"""
    values = []
    for val in signal_data['value']:
        try:
            if isinstance(val, str) and 'b' in val:
                values.append(int(val.replace('b', ''), 2))
            else:
                values.append(float(val))
        except:
            values.append(0.0)
    return np.array(values, dtype=np.float32)


def load_vae_model(model_path=None):
    """Load the trained VAE model"""
    if model_path is None:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'vae_model.pth')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Fix for PyTorch 2.6+ weights_only default
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = SimpleVAE(
        input_dim=checkpoint.get('feature_dim', 128),
        latent_dim=checkpoint.get('latent_dim', 16)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    threshold = checkpoint['threshold']
    
    return model, threshold, device


def verify_verilog_waveform(verilog_code, model_path=None):
    """
    Complete verification function:
    1. Simulate Verilog
    2. Extract features 
    3. Run through VAE
    4. Return anomaly result
    """
    
    # Load VAE
    try:
        model, threshold, device = load_vae_model(model_path)
    except FileNotFoundError:
        return False, 0.0, "VAE model not found. Run vae_mvp.py first!"
    
    # Simulate Verilog
    sim = SimulationService()
    success, stdout, csv_data, error = sim.simulate_verilog(verilog_code)
    
    if not success:
        return False, 0.0, f"Simulation failed: {error}"
    
    # Parse simulation output to CSV if needed
    if not csv_data and stdout:
        lines = stdout.strip().split('\n')
        csv_lines = ['timestamp,signal,value']
        for line in lines:
            if line.startswith('Time='):
                parts = line.split()
                timestamp = None
                for part in parts:
                    if '=' in part:
                        key, val = part.split('=', 1)
                        if key == 'Time':
                            timestamp = val
                        elif timestamp:
                            csv_lines.append(f"{timestamp},{key},{val}")
        csv_data = '\n'.join(csv_lines)
    
    if not csv_data or len(csv_data) < 50:
        return False, 0.0, "No valid waveform data generated"
    
    # Extract features
    features = csv_to_features(csv_data)
    if features is None:
        return False, 0.0, "Could not extract features from waveform"
    
    # Run through VAE
    x = torch.FloatTensor(features).unsqueeze(0).to(device)
    with torch.no_grad():
        x_recon, _, _ = model(x)
        error = F.mse_loss(x, x_recon).item()
    
    is_anomalous = error > threshold
    status = "ANOMALOUS" if is_anomalous else "NORMAL"
    
    return is_anomalous, error, f"{status} (error: {error:.6f}, threshold: {threshold:.6f})"


def test_verification():
    """Test the verification function"""
    
    # Test with a correct design (should be normal)
    correct_verilog = """
`timescale 1ns/1ps
module toggle_ff(input wire clk, rst_n, output reg q);
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) q <= 1'b0;
    else q <= ~q;
end
endmodule

module testbench;
    reg clk, rst_n; wire q;
    toggle_ff dut(.clk(clk), .rst_n(rst_n), .q(q));
    initial begin
        $dumpfile("dump.vcd"); $dumpvars(0, testbench);
        clk = 0; rst_n = 0;
        #20 rst_n = 1;
        #200 $finish;
    end
    always #5 clk = ~clk;
    initial $monitor("Time=%0t clk=%b q=%b", $time, clk, q);
endmodule
"""
    
    # Test with buggy design (should be anomalous)
    buggy_verilog = """
`timescale 1ns/1ps
module bad_toggle(input wire clk, rst_n, output reg q);
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) q <= 1'b1;  // BUG: wrong reset value
    else q <= q;  // BUG: doesn't toggle
end
endmodule

module testbench;
    reg clk, rst_n; wire q;
    bad_toggle dut(.clk(clk), .rst_n(rst_n), .q(q));
    initial begin
        $dumpfile("dump.vcd"); $dumpvars(0, testbench);
        clk = 0; rst_n = 0;
        #20 rst_n = 1;
        #200 $finish;
    end
    always #5 clk = ~clk;
    initial $monitor("Time=%0t clk=%b q=%b", $time, clk, q);
endmodule
"""
    
    print("Testing VAE Verification Function")
    print("=" * 50)
    
    print("\n1. Testing CORRECT design:")
    is_anom, error, msg = verify_verilog_waveform(correct_verilog)
    print(f"   Result: {msg}")
    
    print("\n2. Testing BUGGY design:")
    is_anom, error, msg = verify_verilog_waveform(buggy_verilog)
    print(f"   Result: {msg}")
    
    print("\n" + "=" * 50)
    print("Verification function ready!")


if __name__ == "__main__":
    test_verification()