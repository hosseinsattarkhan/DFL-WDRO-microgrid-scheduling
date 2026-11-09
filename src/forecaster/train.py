import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import sys
import os
import time
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.forecaster.tcn_gmm import TCN_GMM

# ============================================================
# DFL loss (Decision-Focused Loss)
# ============================================================
def dfl_loss(mu, var, pi, true_y, lambda_reg=0.1):
    """
    Compute the DFL loss:
    1. NLL (Negative Log-Likelihood) for forecast accuracy
    2. Regret term (approximated by MSE in this simplified version)
    """
    nll = 0
    for k in range(pi.size(1)):
        p_k = pi[:, k].unsqueeze(-1)
        density = torch.exp(-0.5 * ((true_y - mu[:, k, :]) ** 2) / var[:, k, :]) / torch.sqrt(2 * np.pi * var[:, k, :])
        nll += -torch.sum(torch.log(p_k * density + 1e-8))
    
    nll = nll / true_y.size(0)
    
    expected_mu = torch.sum(pi.unsqueeze(-1) * mu, dim=1)
    regret_proxy = torch.mean((expected_mu - true_y) ** 2)
    
    return nll + lambda_reg * regret_proxy

# ============================================================
# Data preparation
# ============================================================
def create_sequences(data, seq_length=24):
    """Convert time-series data into input/output sequences."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def load_and_prepare_data(file_path="data/processed/dataset_hourly_2010_2012.csv", seq_length=24):
    """Load data and convert to a TCN-friendly format."""
    df = pd.read_csv(file_path, parse_dates=['Timestamp'])
    df = df.set_index('Timestamp')
    
    data = df[['GHI', 'WindSpeed_80m', 'Price']].values
    
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    data = (data - data_min) / (data_max - data_min + 1e-8)
    
    X, Y = create_sequences(data, seq_length)
    
    X = torch.FloatTensor(X).permute(0, 2, 1)
    Y = torch.FloatTensor(Y)
    
    return X, Y, data_min, data_max

# ============================================================
# Training loop
# ============================================================
def train_model(X, Y, epochs=10, batch_size=64, lr=0.001, device='cpu'):
    print(f"Training on {device}...")
    print(f"Number of samples: {len(X)}")
    
    model = TCN_GMM(input_size=3, num_channels=[32, 32, 32], n_components=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            mu, var, pi = model(batch_x)
            loss = dfl_loss(mu, var, pi, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model, losses

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("Loading data...")
    X, Y, data_min, data_max = load_and_prepare_data()
    
    split_idx = int(0.8 * len(X))
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_test, Y_test = X[split_idx:], Y[split_idx:]
    
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    start_time = time.time()
    model, losses = train_model(X_train, Y_train, epochs=10, batch_size=64, device=device)
    total_time = time.time() - start_time
    print(f"\nTraining time: {total_time:.1f} s")
    
    os.makedirs("models/forecaster", exist_ok=True)
    torch.save(model.state_dict(), "models/forecaster/tcn_gmm.pth")
    print("Model saved to models/forecaster/tcn_gmm.pth")
    
    model.eval()
    with torch.no_grad():
        mu, var, pi = model(X_test.to(device))
        test_loss = dfl_loss(mu, var, pi, Y_test.to(device))
        print(f"Test Loss (DFL): {test_loss.item():.4f}")
    
    with torch.no_grad():
        sample_x = X_test[0:1].to(device)
        samples = model.sample(sample_x, n_samples=5)
        print("\nSample predictions (5 samples):")
        print(samples.cpu().numpy())
