import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from matplotlib.lines import Line2D

# Load CSV
csv_path = r"C:\Users\v\Downloads\Data-20251010T152607Z-1-001\Data\labels.csv"
df = pd.read_csv(csv_path)

# Keep only rows where discharge is 0
df = df[df['discharge'] == 0].reset_index(drop=True)

# Ensure numeric
df['time'] = pd.to_numeric(df['time'], errors='coerce')
df['discharge'] = pd.to_numeric(df['discharge'], errors='coerce')

# Map modes to colors
mode_colors = {
    'L-mode': 'blue',
    'H-mode': 'red',
    'ELM': 'green'
}

# Create line segments
points = np.array([df['time'], df['discharge']]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Map colors to each segment based on the starting point's mode
colors = [mode_colors[mode] for mode in df['mode'][:-1]]

# Create LineCollection
lc = LineCollection(segments, colors=colors, linewidths=2)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.add_collection(lc)
ax.set_xlim(df['time'].min(), df['time'].max())
ax.set_ylim(df['discharge'].min() - 0.1, df['discharge'].max() + 0.1)
ax.set_xlabel('Time')
ax.set_ylabel('Discharge')
ax.set_title('Discharge = 0 Over Time with Mode-Based Color')
ax.grid(True)

# Legend
legend_elements = [Line2D([0], [0], color=c, lw=2, label=m) for m, c in mode_colors.items()]
ax.legend(handles=legend_elements)

plt.show()