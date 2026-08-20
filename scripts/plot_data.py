import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Disable macOS font scanner freeze
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

# 1. Load the spreadsheet
data = pd.read_csv('data/pollution_dataset.csv')

# 2. Automatically find the timestamp closest to 0.5 hours
target_time = 0.5
unique_times = data['t'].unique()
closest_time = unique_times[np.abs(unique_times - target_time).argmin()]

snapshot = data[data['t'] == closest_time]

# 3. Draw the color map of the smoke
plt.figure(figsize=(6, 5))
plt.scatter(snapshot['x'], snapshot['y'], c=snapshot['concentration'], cmap='hot', s=15)
plt.colorbar(label='Smoke Concentration')

# 4. Add labels
plt.title(f'Smoke Dispersion at t = {closest_time:.2f} hours')
plt.xlabel('X position (km)')
plt.ylabel('Y position (km)')

# 5. Save the image
plt.savefig('pollution_map.png', bbox_inches='tight')
print(f"Map saved successfully for time t = {closest_time:.2f} hours!")