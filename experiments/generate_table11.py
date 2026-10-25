import torch
import numpy as np
import pandas as pd
import sys
import os
import time
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.forecaster.tcn_gmm import TCN_GMM
from src.forecaster.train import load_and_prepare_data, train_model
from src.data.data_loader import load_scenarios
from models.pyomo.master_ccg import build_master_ccg
import pyomo.environ as pyo

def evaluate_forecaster(model, X_test, Y_test, device='cpu'):
    """ارزیابی forecaster روی داده‌های تست (NLL و MSE)"""
    model.eval()
    with torch.no_grad():
        mu, var, pi = model(X_test.to(device))
        nll = 0
        for k in range(pi.size(1)):
            p_k = pi[:, k].unsqueeze(-1)
            density = torch.exp(-0.5 * ((Y_test.to(device) - mu[:, k, :]) ** 2) / var[:, k, :]) / torch.sqrt(2 * np.pi * var[:, k, :])
            nll += -torch.sum(torch.log(p_k * density + 1e-8))
        nll = nll / Y_test.size(0)
        
        expected_mu = torch.sum(pi.unsqueeze(-1) * mu, dim=1)
        mse = torch.mean((expected_mu - Y_test.to(device)) ** 2)
        
    return nll.item(), mse.item()

def compute_decision_regret(model, X_test, Y_test, data_min, data_max, device='cpu'):
    """
    محاسبه Regret تصمیم‌گیری با استفاده از Forecaster
    (مقادیر PV و Wind به [0,1] محدود می‌شوند، قیمت به مقیاس واقعی برمی‌گردد)
    """
    model.eval()
    
    with torch.no_grad():
        sample_x = X_test[0:1].to(device)
        samples = model.sample(sample_x, n_samples=10) # (1, 10, 3)
    
    samples_np = samples.cpu().numpy()
    mean_sample = np.mean(samples_np[0], axis=0)  # (3,) نرمال‌شده (0 تا 1)
    
    # PV و Wind باید بین 0 و 1 باشند (چون مدل Master انتظار دارد)
    pv_norm = np.clip(mean_sample[0], 0, 1)
    wind_norm = np.clip(mean_sample[1], 0, 1)
    
    # قیمت را به مقیاس واقعی برمی‌گردانیم (0 تا 1 -> 0 تا 100 دلار)
    price_real = mean_sample[2] * (data_max[2] - data_min[2]) + data_min[2]
    price_real = max(price_real, 1.0)
    
    print(f"  مقادیر تصمیم: PV={pv_norm:.2f}, Wind={wind_norm:.2f}, Price={price_real:.2f} $")
    
    # ساخت سناریوی 24 ساعته (تکرار مقادیر در 24 ساعت)
    # قیمت را بر 100 تقسیم می‌کنیم تا مقیاس آن با مدل هماهنگ شود
    scenario_24h = {
        'pv': [float(pv_norm)] * 24,
        'wind': [float(wind_norm)] * 24,
        'price': [float(price_real / 100.0)] * 24
    }
    
    master_model = build_master_ccg([scenario_24h])
    solver = pyo.SolverFactory('appsi_highs')
    solver.config.load_solution = False
    results = solver.solve(master_model, tee=False)
    
    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        return np.inf
    
    return abs(pyo.value(master_model.objective))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"دستگاه: {device}")
    
    print("بارگذاری داده‌ها...")
    X, Y, data_min, data_max = load_and_prepare_data()
    split_idx = int(0.8 * len(X))
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_test, Y_test = X[split_idx:], Y[split_idx:]
    
    results = []
    
    # ============================================================
    # رویکرد 1: Sequential (MSE)
    # ============================================================
    print("\n--- رویکرد 1: Sequential (MSE) ---")
    
    model_mse = TCN_GMM(input_size=3, num_channels=[32, 32, 32], n_components=3).to(device)
    optimizer = torch.optim.Adam(model_mse.parameters(), lr=0.001)
    
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    for epoch in range(5):
        model_mse.train()
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            mu, var, pi = model_mse(batch_x)
            loss = 0
            for k in range(pi.size(1)):
                p_k = pi[:, k].unsqueeze(-1)
                density = torch.exp(-0.5 * ((batch_y - mu[:, k, :]) ** 2) / var[:, k, :]) / torch.sqrt(2 * np.pi * var[:, k, :])
                loss += -torch.sum(torch.log(p_k * density + 1e-8))
            loss = loss / batch_y.size(0)
            loss.backward()
            optimizer.step()
    
    nll_mse, mse_mse = evaluate_forecaster(model_mse, X_test, Y_test, device)
    regret_mse = compute_decision_regret(model_mse, X_test, Y_test, data_min, data_max, device)
    
    results.append({
        'Approach': 'Sequential (MSE)',
        'NLL': nll_mse,
        'MSE': mse_mse,
        'Decision Cost': regret_mse,
    })
    
    # ============================================================
    # رویکرد 2: Proposed (DFL)
    # ============================================================
    print("\n--- رویکرد 2: Proposed (DFL) ---")
    
    model_dfl, _ = train_model(X_train, Y_train, epochs=5, batch_size=64, device=device)
    
    nll_dfl, mse_dfl = evaluate_forecaster(model_dfl, X_test, Y_test, device)
    regret_dfl = compute_decision_regret(model_dfl, X_test, Y_test, data_min, data_max, device)
    
    results.append({
        'Approach': 'Proposed (DFL)',
        'NLL': nll_dfl,
        'MSE': mse_dfl,
        'Decision Cost': regret_dfl,
    })
    
    # ============================================================
    # رویکرد 3: Oracle (بهترین حالت ممکن)
    # ============================================================
    print("\n--- رویکرد 3: Oracle ---")
    
    # Oracle: استفاده از داده‌های واقعی (میانگین Y_test)
    Y_test_np = Y_test.numpy()  # نرمال‌شده (0 تا 1)
    mean_real = np.mean(Y_test_np, axis=0)  # (3,)
    
    pv_oracle = np.clip(mean_real[0], 0, 1)
    wind_oracle = np.clip(mean_real[1], 0, 1)
    
    price_oracle_real = mean_real[2] * (data_max[2] - data_min[2]) + data_min[2]
    price_oracle_real = max(price_oracle_real, 1.0)
    
    print(f"  Oracle (واقعی): PV={pv_oracle:.2f}, Wind={wind_oracle:.2f}, Price={price_oracle_real:.2f} $")
    
    oracle_scenario = {
        'pv': [float(pv_oracle)] * 24,
        'wind': [float(wind_oracle)] * 24,
        'price': [float(price_oracle_real / 100.0)] * 24
    }
    
    master_oracle = build_master_ccg([oracle_scenario])
    solver = pyo.SolverFactory('appsi_highs')
    solver.config.load_solution = False
    res = solver.solve(master_oracle, tee=False)
    
    oracle_cost = abs(pyo.value(master_oracle.objective)) if res.solver.termination_condition == pyo.TerminationCondition.optimal else np.inf
    
    results.append({
        'Approach': 'Oracle (Perfect Info)',
        'NLL': 0.0,
        'MSE': 0.0,
        'Decision Cost': oracle_cost,
    })
    
    # ============================================================
    # ساخت Table 11
    # ============================================================
    df = pd.DataFrame(results)
    
    # محاسبه Regret (%) نسبت به Oracle
    oracle_cost_val = results[-1]['Decision Cost']
    df['Regret (%)'] = (df['Decision Cost'] - oracle_cost_val) / oracle_cost_val * 100
    
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/table11_dfl_comparison.csv", index=False)
    
    print("\n" + "="*70)
    print("Table 11 (DFL Comparison):")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)

if __name__ == "__main__":
    main()
