import numpy as np
import pandas as pd
import os

def generate_advection_diffusion_data():
    # Grid domain parameters
    x = np.linspace(0, 2, 50)  # Spatial X: 0 to 2 km
    y = np.linspace(0, 2, 50)  # Spatial Y: 0 to 2 km
    t = np.linspace(0, 1, 20)  # Time: 0 to 1 hour
    
    X, Y, T = np.meshgrid(x, y, t, indexing='ij')
    
    # Wind vector constants (Advection velocities)
    u_wind = 0.5  # X-direction wind speed
    v_wind = 0.3  # Y-direction wind speed
    D = 0.05      # Diffusion coefficient
    
    # Analytical solution for advection-diffusion from a point source at (0.5, 0.5)
    x0, y0 = 0.5, 0.5
    sigma2 = 4 * D * (T + 0.1)
    
    concentration = (1.0 / (np.pi * sigma2)) * np.exp(
        -((X - x0 - u_wind * T)**2 + (Y - y0 - v_wind * T)**2) / sigma2
    )
    
    # Flatten domain into tabular dataset
    df = pd.DataFrame({
        'x': X.flatten(),
        'y': Y.flatten(),
        't': T.flatten(),
        'concentration': concentration.flatten()
    })
    
    # Save dataset to data/ folder
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/pollution_dataset.csv', index=False)
    print(f"Dataset successfully generated with {len(df)} spatio-temporal data points in data/pollution_dataset.csv!")

if __name__ == '__main__':
    generate_advection_diffusion_data()