import pandas as pd
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from model import PollutionPINN

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def evaluate():
    data = pd.read_csv('data/pollution_dataset.csv')
    
    x = torch.tensor(data['x'].values, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(data['y'].values, dtype=torch.float32).unsqueeze(1)
    t = torch.tensor(data['t'].values, dtype=torch.float32).unsqueeze(1)
    c_true = data['concentration'].values

    # Load saved models
    baseline = PollutionPINN()
    baseline.load_state_dict(torch.load('models/baseline_model.pt'))
    baseline.eval()

    pinn = PollutionPINN()
    pinn.load_state_dict(torch.load('models/pinn_model.pt'))
    pinn.eval()

    with torch.no_grad():
        c_pred_base = baseline(x, y, t).numpy().flatten()
        c_pred_pinn = pinn(x, y, t).numpy().flatten()

    # Calculate metrics across full 50,000-point domain
    r2_base = compute_r2(c_true, c_pred_base)
    r2_pinn = compute_r2(c_true, c_pred_pinn)
    mse_base = np.mean((c_true - c_pred_base) ** 2)
    mse_pinn = np.mean((c_true - c_pred_pinn) ** 2)

    print(f"--- Full Domain Evaluation (50,000 points) ---")
    print(f"Baseline NN -> MSE: {mse_base:.6f} | R2 Score: {r2_base:.4f}")
    print(f"PINN Model  -> MSE: {mse_pinn:.6f} | R2 Score: {r2_pinn:.4f}")

    # Plot visual comparison at t = 0.5 hours
    target_time = 0.5
    unique_times = data['t'].unique()
    closest_time = unique_times[np.abs(unique_times - target_time).argmin()]
    idx = data['t'] == closest_time

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Ground Truth
    im0 = axes[0].scatter(data.loc[idx, 'x'], data.loc[idx, 'y'], c=c_true[idx], cmap='hot', s=10)
    axes[0].set_title('Ground Truth')
    fig.colorbar(im0, ax=axes[0])

    # Baseline Model
    im1 = axes[1].scatter(data.loc[idx, 'x'], data.loc[idx, 'y'], c=c_pred_base[idx], cmap='hot', s=10)
    axes[1].set_title(f'Baseline NN (R2: {r2_base:.2f})')
    fig.colorbar(im1, ax=axes[1])

    # PINN Model
    im2 = axes[2].scatter(data.loc[idx, 'x'], data.loc[idx, 'y'], c=c_pred_pinn[idx], cmap='hot', s=10)
    axes[2].set_title(f'PINN Model (R2: {r2_pinn:.2f})')
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300)
    print("Comparison heatmap saved to comparison_results.png!")

if __name__ == '__main__':
    evaluate()