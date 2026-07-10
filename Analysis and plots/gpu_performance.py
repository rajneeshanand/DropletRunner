#!/usr/bin/env python3

# GPU Training Performance Comparison. Horizontal bar chart showing training time for 50K steps across different GPUs.

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})


gpus = [
    'NVIDIA A40\n(Sol HPC, ima40-gpu)',
    'NVIDIA L40S\n(sol HPC, lake-gpu)',
    'NVIDIA RTX 5060\n(Laptop, 8 GB VRAM)',
]
times_min = [85, 46, 146]
times_hr = [t/60 for t in times_min]

colors = ['#264653', '#2A9D8F', '#E76F51']

BG = 'white'


fig, ax = plt.subplots(figsize=(7, 3.5))
ax.set_facecolor(BG)

y_pos = np.arange(len(gpus))
bars = ax.barh(y_pos, times_min, height=0.55, color=colors, alpha=0.88,
               edgecolor='none', zorder=3)

#Time labels inside/outside bars
for bar, t_min, t_hr in zip(bars, times_min, times_hr):
    # Label inside bar
    if t_min > 60:
        ax.text(bar.get_width() - 4, bar.get_y() + bar.get_height()/2,
                f'{t_min} min ({t_hr:.1f} h)', ha='right', va='center',
                fontsize=11, fontweight='bold', color='white')
    else:
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
                f'{t_min} min ({t_hr:.1f} h)', ha='left', va='center',
                fontsize=11, fontweight='bold', color='#333333')

fastest_idx = np.argmin(times_min)
ax.text(times_min[fastest_idx] + 3, fastest_idx + 0.22, 'fastest',
        fontsize=9, color='#2D6A4F', fontweight='bold', va='center')

ax.set_yticks(y_pos)
ax.set_yticklabels(gpus)
ax.set_xlabel('Wall-clock training time for 50K steps (minutes)')
ax.set_xlim(0, 175)
ax.invert_yaxis()


plt.savefig('fig_gpu_performance.png', bbox_inches='tight',
            pad_inches=0.15, dpi=600)

# Save as PDF for high-quality printing                 # Not necessary because this image will go in supplementary materials
# plt.savefig('fig_gpu_performance.pdf', bbox_inches='tight', pad_inches=0.15, dpi=600)
plt.close()