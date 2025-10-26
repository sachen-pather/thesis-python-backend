# services/vae_waveform_gpu.py
"""
GPU-Accelerated VAE for Waveform Verification
Minimal working implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from io import StringIO
from typing import Optional
from dataclasses import dataclass


def get_device(force_cpu=False):
    """Get best available device"""
    if force_cpu:
        return torch.device('cpu')
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return device
    else:
        print("⚠️ No GPU detected, using CPU")
        return torch.device('cpu')


def get_gpu_memory_info():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1e9,
            'reserved': torch.cuda.memory_reserved() / 1e9,
            'max_allocated': torch.cuda.max_memory_allocated() / 1e9
        }
    return None


class WaveformVAE(nn.Module):
    """VAE model for waveform verification"""
    
    def __init__(self, input_dim=128, hidden_dim=64, latent_dim=16):
        super(WaveformVAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def compute_loss(self, x, x_recon, mu, logvar, beta=1.0):
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return recon_loss + beta * kl_loss, recon_loss, kl_loss


@dataclass
class VerificationResult:
    is_anomalous: bool
    reconstruction_error: float
    anomaly_score: float
    confidence: float
    threshold: float
    details: dict


class GPUWaveformVAEService:
    """GPU-accelerated VAE service"""
    
    def __init__(self, input_dim=128, hidden_dim=64, latent_dim=16, use_gpu=True, device=None):
        if device is not None:
            self.device = device
        else:
            self.device = get_device(force_cpu=not use_gpu)
        
        self.model = WaveformVAE(input_dim, hidden_dim, latent_dim).to(self.device)
        self.threshold = None
        self.is_trained = False
        self.training_history = []
        self.feature_dim = input_dim
        
        print(f"\n{'='*70}")
        print(f"GPU VAE Initialized")
        print(f"Device: {self.device}")
        print(f"Architecture: {input_dim} → {hidden_dim} → {latent_dim}")
        print(f"{'='*70}\n")
    
    def csv_to_features(self, csv_data: str) -> np.ndarray:
        """Convert CSV to features"""
        df = pd.read_csv(StringIO(csv_data))
        signals = sorted(df['signal'].unique())
        features_per_signal = self.feature_dim // max(len(signals), 1)
        
        all_features = []
        for signal in signals:
            signal_data = df[df['signal'] == signal].sort_values('timestamp')
            values = []
            for val in signal_data['value']:
                try:
                    if isinstance(val, str) and 'b' in val:
                        numeric = int(val.replace('b', ''), 2)
                    else:
                        numeric = float(val)
                    values.append(numeric)
                except:
                    values.append(0)
            
            values = np.array(values, dtype=np.float32)
            if len(values) > features_per_signal:
                values = values[:features_per_signal]
            elif len(values) < features_per_signal:
                values = np.pad(values, (0, features_per_signal - len(values)))
            
            if values.max() > 0:
                values = values / values.max()
            
            all_features.extend(values)
        
        features = np.array(all_features, dtype=np.float32)
        if len(features) > self.feature_dim:
            features = features[:self.feature_dim]
        elif len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
        
        return features
    
    def train(self, X_train, epochs=100, batch_size=32, lr=0.001, beta=1.0, verbose=True):
        """Train VAE"""
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        
        if verbose:
            print(f"Training on {len(X_train)} samples for {epochs} epochs...")
        
        import time
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_losses = []
            for (batch_x,) in dataloader:
                x_recon, mu, logvar = self.model(batch_x)
                loss, recon_loss, kl_loss = self.model.compute_loss(batch_x, x_recon, mu, logvar, beta)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {np.mean(epoch_losses):.4f}")
        
        # Set threshold
        self.model.eval()
        errors = []
        with torch.no_grad():
            for i in range(len(X_train)):
                x = torch.FloatTensor(X_train[i:i+1]).to(self.device)
                x_recon, _, _ = self.model(x)
                error = F.mse_loss(x, x_recon).item()
                errors.append(error)
        
        self.threshold = np.mean(errors) + 2 * np.std(errors)
        self.is_trained = True
        
        if verbose:
            print(f"\n✓ Training complete in {time.time()-start_time:.2f}s")
            print(f"Threshold: {self.threshold:.6f}")
    
    def verify_waveform(self, csv_data: str) -> VerificationResult:
        """Verify waveform"""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        features = self.csv_to_features(csv_data)
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            x_recon, _, _ = self.model(x)
            error = F.mse_loss(x, x_recon).item()
        
        is_anomalous = error > self.threshold
        anomaly_score = error / self.threshold
        confidence = min(1.0, abs(error - self.threshold) / self.threshold)
        
        return VerificationResult(
            is_anomalous=is_anomalous,
            reconstruction_error=error,
            anomaly_score=anomaly_score,
            confidence=confidence,
            threshold=self.threshold,
            details={
                'status': 'ANOMALOUS' if is_anomalous else 'NORMAL',
                'message': f"{'⚠️ Anomalous' if is_anomalous else '✓ Normal'} (error: {error:.6f})"
            }
        )
    
    def save_model(self, filepath='vae_gpu_model.pth'):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'threshold': self.threshold,
            'input_dim': self.model.input_dim,
            'hidden_dim': self.model.hidden_dim,
            'latent_dim': self.model.latent_dim
        }, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath='vae_gpu_model.pth'):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model = WaveformVAE(
            checkpoint['input_dim'],
            checkpoint['hidden_dim'],
            checkpoint['latent_dim']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.threshold = checkpoint['threshold']
        self.is_trained = True
        print(f"✓ Model loaded from {filepath}")
    
    def get_device_info(self):
        """Get device info"""
        info = {'device': str(self.device), 'device_type': self.device.type}
        if torch.cuda.is_available():
            info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'cuda_version': torch.version.cuda
            })
        return info