# tests/vae/vae_mvp.py
"""
VAE MVP - Simple, working solution
Uses only verified data from your generator
"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from io import StringIO
import pickle


def get_device():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return torch.device('cuda')
    return torch.device('cpu')


class SimpleVAE(nn.Module):
    def __init__(self, input_dim=128, latent_dim=16):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim), nn.Sigmoid()
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

# Placeholder functions - implement based on your specific needs
def find_clock_edges(clock_data):
    return []

def count_changes_per_clock_cycle(signal_data, clock_edges):
    return 0.0

def analyze_setup_hold_timing(signal_data, clock_edges):
    return 0.0

def measure_clock_synchronization(signal_data, clock_edges):
    return 0.0

def detect_counter_pattern(df):
    return 'count' in str(df['signal'].values).lower()

def detect_logic_gate_pattern(df):
    signals = df['signal'].unique()
    return len(signals) <= 4 and any('out' in str(s).lower() or 'y' in str(s).lower() for s in signals)

def detect_sequential_pattern(df):
    return 'clk' in str(df['signal'].values).lower()

def measure_counting_linearity(df):
    return 0.0

def detect_counter_overflow(df):
    return 0.0

def measure_count_step_size(df):
    return 0.0

def analyze_truth_table_compliance(df):
    return 0.0

def measure_propagation_consistency(df):
    return 0.0

def detect_logic_inversions(df):
    return 0.0

def analyze_state_persistence(df):
    return 0.0

def measure_reset_behavior(df):
    return 0.0

def detect_state_machine_violations(df):
    return 0.0

def main():
    print("\n" + "="*60)
    print("VAE MVP - SIMPLE WORKING VERSION")
    print("="*60)
    
    device = get_device()
    model = SimpleVAE().to(device)
    
    # Load verified training data
    try:
        with open('curated_training_data.pkl', 'rb') as f:
            verified_designs = pickle.load(f)
        print(f"Loaded {len(verified_designs)} verified designs")
    except FileNotFoundError:
        print("No verified_training_data.pkl found!")
        print("Run: python generate_verified_training_data.py first")
        return
    
    # Convert to features
    training_features = []
    for design in verified_designs:
        features = csv_to_features(design['csv'])
        if features is not None:
            training_features.append(features)
    
    if len(training_features) < 10:
        print(f"Only {len(training_features)} valid examples - need at least 10!")
        return
    
    X_train = np.array(training_features)
    print(f"Training on {len(training_features)} examples")
    
    # Train VAE
    X_tensor = torch.FloatTensor(X_train).to(device)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    print("Training...")
    for epoch in range(50):
        losses = []
        for (batch,) in loader:
            x_recon, mu, logvar = model(batch)
            recon = F.mse_loss(x_recon, batch)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + 0.1 * kl
            
            if not torch.isnan(loss):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        
        if losses and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss {np.mean(losses):.4f}")
    
    # Calculate threshold
    model.eval()
    errors = []
    valid_count = 0
    
    with torch.no_grad():
        for x in X_train:
            x_t = torch.FloatTensor(x).unsqueeze(0).to(device)
            x_recon, _, _ = model(x_t)
            error = F.mse_loss(x_t, x_recon).item()
            
            # Only include valid (non-nan) errors
            if not np.isnan(error) and not np.isinf(error):
                errors.append(error)
                valid_count += 1
    
    if len(errors) == 0:
        print("ERROR: No valid errors calculated! Check feature extraction.")
        threshold = 0.1  # Fallback threshold
    else:
        # Calculate threshold with valid errors only
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        threshold = 0.10  # More sensitive threshold
        
        # Sanity check
        if np.isnan(threshold) or np.isinf(threshold):
            threshold = np.max(errors) * 1.1  # 110% of max error as fallback
    
    print(f"\nValid errors: {valid_count}/{len(X_train)}")
    print(f"Error stats: mean={np.mean(errors):.6f}, std={np.std(errors):.6f}")
    print(f"Error range: [{np.min(errors):.6f}, {np.max(errors):.6f}]")
    print(f"Mean ± Std: {mean_error:.6f} ± {std_error:.6f}")
    print(f"Percentiles: 50th={np.percentile(errors, 50):.6f}, 75th={np.percentile(errors, 75):.6f}")
    print(f"Threshold: {threshold:.6f}")
    
    # Test function
    def verify_waveform(csv_data):
        features = csv_to_features(csv_data)
        if features is None:
            return False, 0.0, "Could not extract features"
        
        x = torch.FloatTensor(features).unsqueeze(0).to(device)
        with torch.no_grad():
            x_recon, _, _ = model(x)
            error = F.mse_loss(x, x_recon).item()
        
        is_anomalous = error > threshold
        return is_anomalous, error, f"{'ANOMALOUS' if is_anomalous else 'NORMAL'} (error: {error:.6f})"
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'threshold': threshold,
        'feature_dim': 128,
        'latent_dim': 16
    }, 'vae_model.pth')
    
    print("\n" + "="*60)
    print("VAE MVP COMPLETE!")
    print(f"Model saved to: vae_model.pth")
    print(f"Trained on: {len(training_features)} verified examples")
    print(f"Threshold: {threshold:.6f}")
    print("="*60)
    
    # Test on a few training examples
    print("\nTesting on training data:")
    for i in range(min(3, len(verified_designs))):
        is_anom, error, msg = verify_waveform(verified_designs[i]['csv'])
        print(f"  Example {i+1}: {msg}")


if __name__ == "__main__":
    main()