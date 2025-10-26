# tests/vae/standalone_vae_demo.py
"""
Standalone GPU VAE Demo - FINAL FIXED VERSION
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from io import StringIO
import subprocess
import tempfile
import os


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        return device
    else:
        print("⚠️ Using CPU")
        return torch.device('cpu')


class WaveformVAE(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder with batch norm
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder with batch norm
        self.dec1 = nn.Linear(latent_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.dec2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.dec3 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        h = F.relu(self.bn1(self.enc1(x)))
        h = F.relu(self.bn2(self.enc2(h)))
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10, max=10)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode(self, z):
        h = F.relu(self.bn3(self.dec1(z)))
        h = F.relu(self.bn4(self.dec2(h)))
        return torch.sigmoid(self.dec3(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def compute_loss(self, x, x_recon, mu, logvar, beta=0.1):
        # Reconstruction loss with epsilon for stability
        recon = F.mse_loss(x_recon, x, reduction='mean')
        
        # KL divergence
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon + beta * kl, recon, kl


def simulate_verilog(verilog_code):
    with tempfile.TemporaryDirectory() as tmpdir:
        verilog_path = os.path.join(tmpdir, "design.v")
        with open(verilog_path, "w") as f:
            f.write(verilog_code)
        
        try:
            subprocess.run(
                ["iverilog", "-o", os.path.join(tmpdir, "sim"), verilog_path],
                capture_output=True, timeout=10, check=True
            )
        except:
            return None
        
        try:
            result = subprocess.run(
                ["vvp", os.path.join(tmpdir, "sim")],
                capture_output=True, timeout=10, text=True
            )
        except:
            return None
        
        lines = result.stdout.split('\n')
        csv_lines = ["timestamp,signal,value"]
        
        for line in lines:
            if line.startswith('Time='):
                parts = line.split()
                timestamp = None
                for part in parts:
                    if part.startswith('Time='):
                        timestamp = part.split('=')[1]
                    elif '=' in part and 'Time' not in part:
                        sig, val = part.split('=')
                        if timestamp:
                            csv_lines.append(f"{timestamp},{sig},{val}")
        
        if len(csv_lines) > 1:
            return "\n".join(csv_lines)
        return None


def csv_to_features(csv_data, feature_dim=128):
    df = pd.read_csv(StringIO(csv_data))
    signals = sorted(df['signal'].unique())
    features_per_signal = feature_dim // max(len(signals), 1)
    
    all_features = []
    for signal in signals:
        signal_data = df[df['signal'] == signal].sort_values('timestamp')
        values = []
        for val in signal_data['value']:
            try:
                if isinstance(val, str) and 'b' in val:
                    values.append(int(val.replace('b', ''), 2))
                else:
                    values.append(float(val))
            except:
                values.append(0)
        
        values = np.array(values, dtype=np.float32)
        if len(values) > features_per_signal:
            values = values[:features_per_signal]
        else:
            values = np.pad(values, (0, features_per_signal - len(values)))
        
        # Normalize
        max_val = values.max()
        if max_val > 0:
            values = values / max_val
        
        all_features.extend(values)
    
    features = np.array(all_features, dtype=np.float32)
    if len(features) > feature_dim:
        features = features[:feature_dim]
    else:
        features = np.pad(features, (0, feature_dim - len(features)))
    
    return features


def main():
    print("\n" + "="*70)
    print("STANDALONE GPU VAE DEMO - FINAL FIXED VERSION")
    print("="*70)
    
    device = get_device()
    model = WaveformVAE(input_dim=128, hidden_dim=64, latent_dim=16).to(device)
    
    # Initialize weights properly
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    print(f"✓ Model initialized on {device}")
    
    # Generate training data
    print("\n📊 Phase 1: Generating Training Data")
    print("-"*70)
    
    training_features = []
    
    for width in [2, 3, 4]:
        for _ in range(10):
            verilog = f"""
`timescale 1ns/1ps
module counter(input wire clk, rst, output reg [{width-1}:0] count);
always @(posedge clk or posedge rst) begin
    if (rst) count <= {width}'b0;
    else count <= count + 1'b1;
end
endmodule

module testbench;
    reg clk, rst; wire [{width-1}:0] count;
    counter dut(.clk(clk), .rst(rst), .count(count));
    initial begin
        $dumpfile("dump.vcd"); $dumpvars(0, testbench);
        clk = 0; rst = 1; #20 rst = 0; #150 $finish;
    end
    always #5 clk = ~clk;
    initial $monitor("Time=%0t clk=%b rst=%b count=%d", $time, clk, rst, count);
endmodule
"""
            csv_data = simulate_verilog(verilog)
            if csv_data:
                features = csv_to_features(csv_data)
                training_features.append(features)
                print(f"  Generated {len(training_features)}/30", end="\r")
    
    X_train = np.array(training_features)
    print(f"\n✓ Collected {len(training_features)} training examples")
    
    # Add small noise to prevent zero variance
    X_train = X_train + np.random.randn(*X_train.shape).astype(np.float32) * 0.001
    
    # Train VAE
    print("\n🧠 Phase 2: Training VAE on GPU")
    print("-"*70)
    
    X_tensor = torch.FloatTensor(X_train).to(device)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    model.train()
    import time
    start = time.time()
    
    for epoch in range(100):
        epoch_losses = []
        for (batch_x,) in dataloader:
            optimizer.zero_grad()
            
            x_recon, mu, logvar = model(batch_x)
            loss, recon, kl = model.compute_loss(batch_x, x_recon, mu, logvar, beta=0.1)
            
            # Check for NaN and skip if found
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf at epoch {epoch+1}, skipping batch")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        if epoch_losses and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/100 | Loss: {np.mean(epoch_losses):.4f}")
    
    train_time = time.time() - start
    print(f"\n✓ Training complete in {train_time:.2f}s")
    
    # Calculate threshold
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(len(X_train)):
            x = torch.FloatTensor(X_train[i:i+1]).to(device)
            x_recon, _, _ = model(x)
            error = F.mse_loss(x, x_recon).item()
            if not np.isnan(error) and not np.isinf(error):
                errors.append(error)
    
    if errors:
        threshold = np.mean(errors) + 2 * np.std(errors)
        print(f"Threshold: {threshold:.6f}")
    else:
        threshold = 0.1
        print(f"Using default threshold: {threshold}")
    
    # Test on correct
    print("\n✅ Phase 3: Testing Correct Waveforms")
    print("-"*70)
    
    correct_results = []
    for i in range(5):
        verilog = """
`timescale 1ns/1ps
module counter(input wire clk, rst, output reg [3:0] count);
always @(posedge clk or posedge rst) begin
    if (rst) count <= 4'b0;
    else count <= count + 1'b1;
end
endmodule

module testbench;
    reg clk, rst; wire [3:0] count;
    counter dut(.clk(clk), .rst(rst), .count(count));
    initial begin
        $dumpfile("dump.vcd"); $dumpvars(0, testbench);
        clk = 0; rst = 1; #20 rst = 0; #150 $finish;
    end
    always #5 clk = ~clk;
    initial $monitor("Time=%0t count=%d", $time, count);
endmodule
"""
        csv = simulate_verilog(verilog)
        if csv:
            features = csv_to_features(csv)
            x = torch.FloatTensor(features).unsqueeze(0).to(device)
            with torch.no_grad():
                x_recon, _, _ = model(x)
                error = F.mse_loss(x, x_recon).item()
            if not np.isnan(error):
                is_anomalous = error > threshold
                correct_results.append(is_anomalous)
                status = "✓" if not is_anomalous else "✗"
                print(f"Test {i+1}: {status} | Error: {error:.6f}")
    
    # Test on buggy
    print("\n🐛 Phase 4: Testing Buggy Waveforms")
    print("-"*70)
    
    buggy_designs = [
        ("Wrong Reset", """
`timescale 1ns/1ps
module counter(input wire clk, rst, output reg [3:0] count);
always @(posedge clk or posedge rst) begin
    if (rst) count <= 4'b1111;
    else count <= count + 1'b1;
end
endmodule

module testbench;
    reg clk, rst; wire [3:0] count;
    counter dut(.clk(clk), .rst(rst), .count(count));
    initial begin
        $dumpfile("dump.vcd"); $dumpvars(0, testbench);
        clk = 0; rst = 1; #20 rst = 0; #150 $finish;
    end
    always #5 clk = ~clk;
    initial $monitor("Time=%0t count=%d", $time, count);
endmodule
"""),
        ("Double Increment", """
`timescale 1ns/1ps
module counter(input wire clk, rst, output reg [3:0] count);
always @(posedge clk or posedge rst) begin
    if (rst) count <= 4'b0;
    else count <= count + 2'b10;
end
endmodule

module testbench;
    reg clk, rst; wire [3:0] count;
    counter dut(.clk(clk), .rst(rst), .count(count));
    initial begin
        $dumpfile("dump.vcd"); $dumpvars(0, testbench);
        clk = 0; rst = 1; #20 rst = 0; #150 $finish;
    end
    always #5 clk = ~clk;
    initial $monitor("Time=%0t count=%d", $time, count);
endmodule
"""),
    ]
    
    buggy_results = []
    for name, verilog in buggy_designs:
        csv = simulate_verilog(verilog)
        if csv:
            features = csv_to_features(csv)
            x = torch.FloatTensor(features).unsqueeze(0).to(device)
            with torch.no_grad():
                x_recon, _, _ = model(x)
                error = F.mse_loss(x, x_recon).item()
            if not np.isnan(error):
                is_anomalous = error > threshold
                buggy_results.append(is_anomalous)
                status = "✓ CAUGHT" if is_anomalous else "✗ MISSED"
                print(f"{name:20s}: {status} | Error: {error:.6f}")
    
    # Metrics
    print("\n📊 Phase 5: Metrics")
    print("-"*70)
    
    if buggy_results and correct_results:
        tp = sum(buggy_results)
        tn = sum(not r for r in correct_results)
        fp = sum(correct_results)
        fn = len(buggy_results) - tp
        
        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        print(f"Accuracy:  {acc:.2%}")
        print(f"Precision: {prec:.2%}")
        print(f"Recall:    {rec:.2%}")
        print(f"F1 Score:  {f1:.2%}")
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()