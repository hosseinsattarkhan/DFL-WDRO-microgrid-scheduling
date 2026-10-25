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
# تابع زیان DFL (Decision-Focused Loss)
# ============================================================
def dfl_loss(mu, var, pi, true_y, lambda_reg=0.1):
    """
    محاسبه تابع زیان DFL:
    1. NLL (Negative Log-Likelihood) برای دقت پیش‌بینی
    2. ترم Regret (در این نسخه ساده، از MSE به عنوان تقریب استفاده می‌کنیم)
    """
    # NLL برای هر جزء
    nll = 0
    for k in range(pi.size(1)):
        # احتمال هر جزء
        p_k = pi[:, k].unsqueeze(-1)
        # چگالی نرمال برای هر جزء
        density = torch.exp(-0.5 * ((true_y - mu[:, k, :]) ** 2) / var[:, k, :]) / torch.sqrt(2 * np.pi * var[:, k, :])
        nll += -torch.sum(torch.log(p_k * density + 1e-8))
    
    nll = nll / true_y.size(0)
    
    # ترم Regret (تقریب ساده با MSE)
    # در نسخه واقعی، این ترم از حل مدل Pyomo می‌آید
    expected_mu = torch.sum(pi.unsqueeze(-1) * mu, dim=1)
    regret_proxy = torch.mean((expected_mu - true_y) ** 2)
    
    return nll + lambda_reg * regret_proxy

# ============================================================
# آماده‌سازی داده‌ها
# ============================================================
def create_sequences(data, seq_length=24):
    """تبدیل داده‌های سری زمانی به دنباله‌های ورودی/خروجی"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)] # (seq_len, 3)
        y = data[i + seq_length]      # (3,)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def load_and_prepare_data(file_path="data/processed/dataset_hourly_2010_2012.csv", seq_length=24):
    """بارگذاری داده‌ها و تبدیل به فرمت مناسب برای TCN"""
    df = pd.read_csv(file_path, parse_dates=['Timestamp'])
    df = df.set_index('Timestamp')
    
    # انتخاب ستون‌های مورد نیاز (PV, Wind, Price)
    data = df[['GHI', 'WindSpeed_80m', 'Price']].values
    
    # نرمال‌سازی (Min-Max)
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    data = (data - data_min) / (data_max - data_min + 1e-8)
    
    # ساخت دنباله‌ها
    X, Y = create_sequences(data, seq_length)
    
    # تبدیل به Tensor
    X = torch.FloatTensor(X).permute(0, 2, 1) # (batch, 3, seq_len) برای Conv1d
    Y = torch.FloatTensor(Y)
    
    return X, Y, data_min, data_max

# ============================================================
# حلقه آموزش
# ============================================================
def train_model(X, Y, epochs=10, batch_size=64, lr=0.001, device='cpu'):
    print(f"در حال آموزش روی {device}...")
    print(f"تعداد نمونه‌ها: {len(X)}")
    
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
# اجرای اصلی
# ============================================================
if __name__ == "__main__":
    # تشخیص GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"دستگاه: {device}")
    
    print("بارگذاری داده‌ها...")
    X, Y, data_min, data_max = load_and_prepare_data()
    
    # تقسیم داده‌ها به آموزش و تست (80/20)
    split_idx = int(0.8 * len(X))
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_test, Y_test = X[split_idx:], Y[split_idx:]
    
    print(f"آموزش: {len(X_train)} | تست: {len(X_test)}")
    
    # آموزش
    start_time = time.time()
    model, losses = train_model(X_train, Y_train, epochs=10, batch_size=64, device=device)
    total_time = time.time() - start_time
    print(f"\nزمان آموزش: {total_time:.1f} ثانیه")
    
    # ذخیره مدل
    os.makedirs("models/forecaster", exist_ok=True)
    torch.save(model.state_dict(), "models/forecaster/tcn_gmm.pth")
    print("مدل در models/forecaster/tcn_gmm.pth ذخیره شد.")
    
    # ارزیابی روی تست
    model.eval()
    with torch.no_grad():
        mu, var, pi = model(X_test.to(device))
        test_loss = dfl_loss(mu, var, pi, Y_test.to(device))
        print(f"Test Loss (DFL): {test_loss.item():.4f}")
    
    # پیش‌بینی برای یک نمونه (نمایش)
    with torch.no_grad():
        sample_x = X_test[0:1].to(device)
        samples = model.sample(sample_x, n_samples=5) # (1, 5, 3)
        print("\nنمونه پیش‌بینی (5 نمونه):")
        print(samples.cpu().numpy())
