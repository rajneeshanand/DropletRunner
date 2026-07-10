#!/usr/bin/env python3

#Ablation Study: Training steps vs Success rate


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


# Shared x-axis categories (training configurations)
configs = [
    '50K steps\n50 episodes',
    '50K steps\n100 episodes',
    '150K steps\n100 episodes',
    '450K steps\n100 episodes',
]

# I-shape: only tested with 50 episodes
# So for 100-episode configs, I-shape has no data - use None
I_success = [100, None, 100, 95]   # 150K/50ep=100%, 450K/50ep=95%

# L-shape
L_success = [30, 95, 95, 90]

# Colors
I_COLOR = '#264653'    # dark teal
L_COLOR = '#580F41'    # dark purple
OPTIMAL_EDGE = 'Black'  # for optimal performance annotation
RED = '#C1121F'
BG = '#FAFAFA'


fig, ax = plt.subplots(figsize=(8, 4.5))
ax.set_facecolor(BG)

x = np.arange(len(configs))
bar_width = 0.32
offset = bar_width / 2 + 0.02

# I-shape bars 
I_vals = []
I_x = []
for i, val in enumerate(I_success):
    if val is not None:
        I_vals.append(val)
        I_x.append(x[i] - offset)

bars_I = ax.bar(I_x, I_vals, width=bar_width, color=I_COLOR, alpha=0.88,
                edgecolor='none', linewidth=0.0, zorder=3, label='I-shape (113 mm)')

# L-shape bars 
bars_L = ax.bar(x + offset, L_success, width=bar_width, color=L_COLOR, alpha=0.88,
                edgecolor='none', linewidth=0.0, zorder=3, label='L-shape (173.8 mm)')

# "No data" marker for I-shape at 50K/100ep
ax.text(x[1] - offset, 5, '', ha='center', va='bottom', fontsize=8,
        color='#999999', style='italic')

# Data improvement arrow for L-shape (50ep to 100ep)
ax.annotate('', xy=(x[1] + offset, 93), xytext=(x[0] + offset, 33),
            arrowprops=dict(arrowstyle='->', color=OPTIMAL_EDGE, lw=2.0))
ax.text((x[0] + x[1]) / 2 + offset, 85, '+50 episodes\n30% → 95%',
        ha='center', fontsize=9, color=OPTIMAL_EDGE, fontweight='bold')

# Overtraining annotation (450K)
ax.annotate('overtraining', xy=(x[3] + offset + 0.05, 91), rotation=90, va='center', ha='center',
            xytext=(x[3] + offset + 0.35, 58),
            fontsize=9, color='Black', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='Black', lw=2.0))


ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.set_xlabel('Training configuration')
ax.set_ylabel('Navigation success rate (%)')
ax.set_ylim(0, 112)
ax.set_xlim(-0.5, len(configs) - 0.3)

ax.legend(loc='upper right', frameon=False, fancybox=False,
          facecolor='white', edgecolor='none',
          borderpad=0.2)

plt.savefig('fig_ablation.png', bbox_inches='tight',
            pad_inches=0.15, dpi=600)
plt.close()