
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib

# Set backend for animation
matplotlib.use('Agg')

# path = "/Users/wan410/Documents/VSCode/torch-cfd/data/fnodata_extra_128x128_N2_v1e-3_T50_steps100_alpha2.5_tau7.pt"
# path = "/Users/wan410/Documents/VSCode/torch-cfd/data/McWilliams2d_128x128_N2_Re1000_T100.pt"
# path = "/Users/wan410/Documents/VSCode/torch-cfd/fno/data/McWilliams2d_128x128_N2_Re5000_T100.pt"
path = "/Users/wan410/Documents/VSCode/torch-cfd/fno/data/McWilliams2d_128x128_N1_Re100_T100.pt"
if os.path.exists(path):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f'Failed to load data from {path}: {e}')
        data = torch.load(path)
else:
    with open(path, 'rb') as f:
        data = pickle.load(f)

# var_name = 'vorticity'
# var_name = 'ux'
var_name = 'vy'

# Extract vorticity data: shape (1, 100, 128, 128) -> (128, 128, 100)
vorticity = data[var_name][0].squeeze().numpy()  # Remove batch dimension
vorticity = np.transpose(vorticity, (1, 2, 0))  # (H, W, T) = (128, 128, 100)

print(f"{var_name} shape: {data[var_name].shape}")
print(f"Time steps: {data[var_name].shape[2]}")

# Set up the animation
cmap = 'RdBu_r'
vmin = np.min(vorticity)
vmax = np.max(vorticity)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
img = ax.imshow(vorticity[..., 0], vmin=vmin, vmax=vmax, cmap=cmap)
ax.axis('off')
title = ax.set_title(f'{var_name} T+1')

def update(frame_idx):
    img.set_data(vorticity[..., frame_idx])
    title.set_text(f'{var_name} T+{frame_idx+1}')
    return img, title

# Create animation
anim = FuncAnimation(fig, update, frames=vorticity.shape[2], interval=200, blit=False)

# Save as GIF
name = path.split('/')[-1].split('.')[0]
gif_path = f'{name}_{var_name}.gif'
try:
    anim.save(gif_path, writer=PillowWriter(fps=5))
    print(f"GIF saved as: {gif_path}")
except Exception as e:
    print(f'Failed to save GIF due to: {e}')

# Try MP4 as well if ffmpeg is available
# mp4_path = f'{name}.mp4'
# anim.save(mp4_path, writer='ffmpeg', fps=5)
# print(f"MP4 saved as: {mp4_path}")
# try:

# except Exception as e:
    # print(f'FFmpeg not available or failed to save MP4: {e}')

plt.close(fig)
print("Animation complete!")
