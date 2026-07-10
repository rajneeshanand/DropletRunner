#!/usr/bin/env python3

# Droplet Velocity Profile Plot - 3x3 
#Row 1: I-shape (fast, medium, slow); Row 2: L-shape (fast, medium, slow); Row 3: Arc (fast, medium, slow)



import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

target_cmap = LinearSegmentedColormap.from_list(
    "target_pink_purple",
    [
        "#d7bcb4",  # light beige-pink
        "#c3959a",
        "#b57a8e",
        "#985f82",
        "#754873",
        "#4e325c",
        "#0f0b14"   # black
    ],
    N=256
)


LSHAPE_DIR = 'mbrl_actions_ep20'
ISHAPE_DIR = 'mbrl_actions_Ishape_ep20'
ARC_DIR = 'mbrl_actions_arc'
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ

I_PATH_X = [87.5, -21.5]
I_PATH_Y = [-2.0, -2.0]

L_PATH_X = [87, 87, -26]
L_PATH_Y = [-64, 3, 3]


# For Arc, we can either load from JSON or generate using circle parameters, we chose JSON

import json
if os.path.exists('path_waypoints_mm_ARC.json'):
    with open('path_waypoints_mm_ARC.json') as f:
        arc_wp = np.array(json.load(f))
    ARC_PATH_X = arc_wp[:, 0].tolist()
    ARC_PATH_Y = arc_wp[:, 1].tolist()
else:
    cx, cy, r = 36.24, -68.67, 87.83
    theta = np.linspace(np.radians(19.0), np.radians(154.9), 105)
    ARC_PATH_X = (cx + r * np.cos(theta)).tolist()
    ARC_PATH_Y = (cy + r * np.sin(theta)).tolist()


#L-shape

if not os.path.exists(LSHAPE_DIR):
    print(f"ERROR: Folder '{LSHAPE_DIR}' not found!")
    sys.exit(1)

lshape_eps = {}
for f in sorted(os.listdir(LSHAPE_DIR)):
    if f.endswith('.csv'):
        df = pd.read_csv(os.path.join(LSHAPE_DIR, f))
        n = len(df)
        prog = df['progress'].iloc[-1]
        if prog > 150 and n < 590:
            lshape_eps[f] = {'n': n, 'df': df}

lshape_sorted = sorted(lshape_eps.items(), key=lambda x: x[1]['n'])
print(f"L-shape: loaded {len(lshape_sorted)} successful episodes")


# Select L-shape fast, medium, slow
l_fast_candidates = [e for e in lshape_sorted if 30 < e[1]['n'] < 50]
l_fast = l_fast_candidates[0] if l_fast_candidates else lshape_sorted[0]
l_medium_candidates = [e for e in lshape_sorted if 177 < e[1]['n'] < 180]
l_medium = l_medium_candidates[0] if l_medium_candidates else lshape_sorted[len(lshape_sorted)//2]
l_slow = lshape_sorted[-1]

print(f"  Fast:   {l_fast[0]} ({l_fast[1]['n']} steps)")
print(f"  Medium: {l_medium[0]} ({l_medium[1]['n']} steps)")
print(f"  Slow:   {l_slow[0]} ({l_slow[1]['n']} steps)")


#I-shape

if not os.path.exists(ISHAPE_DIR):
    print(f"ERROR: Folder '{ISHAPE_DIR}' not found!")
    sys.exit(1)

ishape_eps = {}
for f in sorted(os.listdir(ISHAPE_DIR)):
    if f.endswith('.csv'):
        df = pd.read_csv(os.path.join(ISHAPE_DIR, f))
        n = len(df)
        prog = df['progress'].iloc[-1]
        if prog > 50 and n < 590:
            ishape_eps[f] = {'n': n, 'df': df}

ishape_sorted = sorted(ishape_eps.items(), key=lambda x: x[1]['n'])
print(f"\nI-shape: loaded {len(ishape_sorted)} successful episodes")

# Select I-shape fast, medium, slow
i_fast = ishape_sorted[0]
i_medium = ishape_sorted[len(ishape_sorted)//2]
i_slow = ishape_sorted[-1]

print(f"  Fast:   {i_fast[0]} ({i_fast[1]['n']} steps)")
print(f"  Medium: {i_medium[0]} ({i_medium[1]['n']} steps)")
print(f"  Slow:   {i_slow[0]} ({i_slow[1]['n']} steps)")




if not os.path.exists(ARC_DIR):
    print(f"ERROR: Folder '{ARC_DIR}' not found!")
    sys.exit(1)

arc_eps = {}
for f in sorted(os.listdir(ARC_DIR)):
    if f.endswith('.csv'):
        df = pd.read_csv(os.path.join(ARC_DIR, f))
        n = len(df)
        prog = df['progress'].iloc[-1]
        if prog > 150 and n < 590:
            arc_eps[f] = {'n': n, 'df': df}

arc_sorted = sorted(arc_eps.items(), key=lambda x: x[1]['n'])
print(f"\nArc: loaded {len(arc_sorted)} successful episodes")

print("\n=== Arc episodes (sorted by steps) ===")
for i, (fname, ep) in enumerate(arc_sorted):
    print(f"  [{i}] {fname}: {ep['n']} steps ({ep['n']*DT:.1f}s), progress={ep['df']['progress'].iloc[-1]:.1f}mm")

# Select Arc fast, medium, slow — adjust step ranges after checking printed list above
a_fast_candidates = [e for e in arc_sorted if 50 < e[1]['n'] < 60]
a_fast = a_fast_candidates[0] if a_fast_candidates else arc_sorted[0]
a_medium_candidates = [e for e in arc_sorted if 100 < e[1]['n'] < 120]
a_medium = a_medium_candidates[0] if a_medium_candidates else arc_sorted[len(arc_sorted)//2]
a_slow = arc_sorted[-1]

print(f"  Fast:   {a_fast[0]} ({a_fast[1]['n']} steps)")
print(f"  Medium: {a_medium[0]} ({a_medium[1]['n']} steps)")
print(f"  Slow:   {a_slow[0]} ({a_slow[1]['n']} steps)")



plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

#fig, axes = plt.subplots(2, 3, figsize=(15, 7))
fig, axes = plt.subplots(3, 3, figsize=(15, 7), gridspec_kw={'height_ratios': [1, 1, 1]})
plt.subplots_adjust(wspace=0.8, hspace=0.2)

vmin, vmax = 0, 175

panels = [
    # Row 0: I-shape
    #(axes[0, 0], i_fast, f'I-shape: Fast ({i_fast[1]["n"]} steps, {i_fast[1]["n"]*DT:.1f}s)', 'a', I_PATH_X, I_PATH_Y),
    (axes[0, 0], i_fast, f'I-shape: Fast ({i_fast[1]["n"]*DT:.1f}s)',
     'a', I_PATH_X, I_PATH_Y),
    (axes[0, 1], i_medium, f'I-shape: Medium ({i_medium[1]["n"]*DT:.1f}s)',
     'b', I_PATH_X, I_PATH_Y),
    (axes[0, 2], i_slow, f'I-shape: Slow ({i_slow[1]["n"]*DT:.1f}s)',
     'c', I_PATH_X, I_PATH_Y),

    # Row 1: L-shape
    (axes[1, 0], l_fast, f'L-shape: Fast ({l_fast[1]["n"]*DT:.1f}s)',
     'd', L_PATH_X, L_PATH_Y),
    (axes[1, 1], l_medium, f'L-shape: Medium ({l_medium[1]["n"]*DT:.1f}s)',
     'e', L_PATH_X, L_PATH_Y),
    (axes[1, 2], l_slow, f'L-shape: Slow ({l_slow[1]["n"]*DT:.1f}s)',
     'f', L_PATH_X, L_PATH_Y),


     # Row 2: Arc
    (axes[2, 0], a_fast, f'Arc: Fast ({a_fast[1]["n"]*DT:.1f}s)',
     'g', ARC_PATH_X, ARC_PATH_Y),
    (axes[2, 1], a_medium, f'Arc: Medium ({a_medium[1]["n"]*DT:.1f}s)',
     'h', ARC_PATH_X, ARC_PATH_Y),
    (axes[2, 2], a_slow, f'Arc: Slow ({a_slow[1]["n"]*DT:.1f}s)',
     'i', ARC_PATH_X, ARC_PATH_Y),
]

for ax, (fname, ep), title, panel_label, path_x, path_y in panels:
    df = ep['df']
    x = df['x_mm'].values
    y = df['y_mm'].values

    # Compute velocity
    dx = np.diff(x) / DT
    dy = np.diff(y) / DT
    speed = np.sqrt(dx**2 + dy**2)
    speed_clipped = np.clip(speed, 0, np.percentile(speed, 95))

    # Draw path
    for i in range(len(path_x) - 1):
        ax.plot([path_x[i], path_x[i+1]], [path_y[i], path_y[i+1]],
                '-', color='gray', linewidth=5, alpha=0.3)

    # Scatter colored by velocity
    sc = ax.scatter(x[1:], y[1:], c=speed_clipped, cmap=target_cmap, s=60,
                    edgecolors='none', linewidths=0.1, zorder=5,
                    vmin=vmin, vmax=vmax)

    # Start marker (green circle)
    ax.plot(x[0], y[0], 'o', color='green', markersize=14, zorder=10)
    # End marker (red star)
    ax.plot(x[-1], y[-1], '*', color='red', markersize=16, zorder=10)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(title)
    #ax.set_aspect('equal')

    if panel_label in ['a', 'b', 'c']:
        ax.set_ylim(-20, 20)
    elif panel_label in ['d', 'e', 'f']:
        ax.set_ylim(-70, 15)
    else:
        ax.set_ylim(-50, 30)
        ax.set_xlim(-55, 135)
        ax.set_aspect('equal')

    # Bold panel label like NMI style
    ax.text(-0.2, 1.22, f'{panel_label}', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top')

plt.tight_layout(w_pad=3.0, h_pad=1.0)
cbar = fig.colorbar(sc, ax=axes, shrink=1.0, pad=0.06)
cbar.set_label('Velocity (mm/s)', fontsize=16)

plt.savefig('fig_velocity_profile_3x3.png', bbox_inches='tight', dpi=600)

plt.savefig('fig_velocity_profile_3x3.pdf', bbox_inches='tight', dpi=600)