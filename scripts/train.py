import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from model import PollutionPINN

torch.manual_seed(42)
np.random.seed(42)

# Load dataset
data = pd.read_csv('data/pollution_dataset.csv')

# Subsample sparse sensor readings (200 spatial-temporal points)
n_sensor_samples = 200
sample_indices = np.random.choice(len(data), size=n_sensor_samples, replace=False)
sparse_data = data.iloc[sample_indices]

x_sensor = torch.tensor(sparse_data['x'].values, dtype=torch.float32).unsqueeze(1)
y_sensor = torch.tensor(sparse_data['y'].values, dtype=torch.float32).unsqueeze(1)
t_sensor = torch.tensor(sparse_data['t'].values, dtype=torch.float32).unsqueeze(1)
c_sensor = torch.tensor(sparse_data['concentration'].values, dtype=torch.float32).unsqueeze(1)

# Dense domain sampling for physics constraint evaluation
n_pde_samples = 2000
pde_indices = np.random.choice(len(data), size=n_pde_samples, replace=False)
pde_data = data.iloc[pde_indices]

x_pde = torch.tensor(pde_data['x'].values, dtype=torch.float32, requires_grad=True).unsqueeze(1)
y_pde = torch.tensor(pde_data['y'].values, dtype=torch.float32, requires_grad=True).unsqueeze(1)
t_pde = torch.tensor(pde_data['t'].values, dtype=torch.float32, requires_grad=True).unsqueeze(1)

u_wind, v_wind = 0.5, 0.3
D = 0.05

def compute_pde_residual(model, x, y, t):
    c = model(x, y, t)
    
    # Automatic differentiation for spatial and temporal gradients
    c_t = torch.autograd.grad(c, t, grad_outputs=torch.ones_like(c), create_graph=True)[0]
    c_x = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c), create_graph=True)[0]
    c_y = torch.autograd.grad(c, y, grad_outputs=torch.ones_like(c), create_graph=True)[0]
    
    c_xx = torch.autograd.grad(c_x, x, grad_outputs=torch.ones_like(c_x), create_graph=True)[0]
    c_yy = torch.autograd.grad(c_y, y, grad_outputs=torch.ones_like(c_y), create_graph=True)[0]
    
    residual = c_t + u_wind * c_x + v_wind * c_y - D * (c_xx + c_yy)
    return torch.mean(residual ** 2)

print("--- Training Baseline Neural Network (Pure Data) ---")
baseline_model = PollutionPINN()
optimizer_base = torch.optim.Adam(baseline_model.parameters(), lr=1e-3)
mse_loss = nn.MSELoss()

for epoch in range(1, 1001):
    optimizer_base.zero_grad()
    c_pred = baseline_model(x_sensor, y_sensor, t_sensor)
    loss = mse_loss(c_pred, c_sensor)
    loss.backward()
    optimizer_base.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Sensor Loss: {loss.item():.6f}")

print("\n--- Training Physics-Informed Neural Network (PINN) ---")
pinn_model = PollutionPINN()
optimizer_pinn = torch.optim.Adam(pinn_model.parameters(), lr=1e-3)
pde_weight = 0.1

for epoch in range(1, 1001):
    optimizer_pinn.zero_grad()
    c_pred = pinn_model(x_sensor, y_sensor, t_sensor)
    loss_data = mse_loss(c_pred, c_sensor)
    loss_pde = compute_pde_residual(pinn_model, x_pde, y_pde, t_pde)
    
    total_loss = loss_data + pde_weight * loss_pde
    total_loss.backward()
    optimizer_pinn.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Total Loss: {total_loss.item():.6f} (Data: {loss_data.item():.6f}, Physics: {loss_pde.item():.6f})")

os.makedirs('models', exist_ok=True)
torch.save(baseline_model.state_dict(), 'models/baseline_model.pt')
torch.save(pinn_model.state_dict(), 'models/pinn_model.pt')
print("\nModels successfully saved to models/ directory!")