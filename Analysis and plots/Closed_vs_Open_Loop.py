#!/usr/bin/env python3

# Open-Loop vs Closed-Loop
# Panel (a): trajectory comparison
# Panel (b): Velocity profile (both in one plot, two colormaps)
# Panel (c): Progress along path vs time


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.colorbar as mcbar


# Closed-loop: blue gradient
cl_cmap = LinearSegmentedColormap.from_list(
    "cl_blue",
    ["#CAF0F8", "#90E0EF", "#0077B6", "#023E8A", "#03045E"],
    N=256
)

# Open-loop: red gradient
ol_cmap = LinearSegmentedColormap.from_list(
    "ol_red",
    ["#FFCCD5", "#FFB3C1", "#E63946", "#A4161A", "#660708"],
    N=256
)

cl = pd.read_csv('closed_loop.csv')
ol = pd.read_excel('open_loop.xlsx')

L_PATH_X = [87, 87, -26]
L_PATH_Y = [-64, 3, 3]

DT = 0.05
vmin, vmax = 0, 175

print(f"Closed-loop: {len(cl)} steps, progress={cl['progress'].iloc[-1]:.1f}mm")
print(f"Open-loop:   {len(ol)} sampled points")


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 18,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

BG_COLOR = 'white'
GRID_COLOR = '#E5E5E5'
CL_COLOR = '#0077B6'
OL_COLOR = '#E63946'


fig, axes = plt.subplots(1, 3, figsize=(16, 5))
plt.subplots_adjust(wspace=0.35)

# panel a: Trajectory comparison

ax_a = axes[0]
ax_a.set_facecolor(BG_COLOR)

# L-shape path
for i in range(len(L_PATH_X) - 1):
    ax_a.plot([L_PATH_X[i], L_PATH_X[i+1]], [L_PATH_Y[i], L_PATH_Y[i+1]], '-', color='gray', linewidth=8, alpha=0.25, solid_capstyle='round')

# Closed-loop trajectory
ax_a.plot(cl['x_mm'].values, cl['y_mm'].values,
          color=CL_COLOR, linewidth=1.8, alpha=0.85, zorder=4,
          label=f'Closed-loop ')

# Open-loop trajectory
ax_a.plot(ol['x_mm'].values, ol['y_mm'].values,
          color=OL_COLOR, linewidth=1.8, alpha=0.85, zorder=3,
          linestyle='-', label=f'Open-loop')

# Stuck marker
stuck_x, stuck_y = ol['x_mm'].iloc[-1], ol['y_mm'].iloc[-1]
ax_a.scatter(stuck_x, stuck_y, s=120, c=OL_COLOR, marker='X', edgecolors='white', linewidth=1.2, zorder=8)
ax_a.annotate('stuck at \ncorner', xy=(stuck_x, stuck_y), xytext=(stuck_x-8, stuck_y + 8), 
                fontsize=10, color=OL_COLOR, fontweight='bold', arrowprops=dict(arrowstyle='->', color=OL_COLOR, lw=1.5))

# Start marker
ax_a.plot(cl['x_mm'].iloc[0], cl['y_mm'].iloc[0], 'o', color='#2D6A4F', markersize=12, zorder=10)
ax_a.text(cl['x_mm'].iloc[0] - 5, cl['y_mm'].iloc[0] - 8, 'Start', fontsize=10, color='#2D6A4F', fontweight='bold')

# End marker (closed-loop)
ax_a.plot(cl['x_mm'].iloc[-1], cl['y_mm'].iloc[-1], '*', color=CL_COLOR, markersize=14, zorder=10)

ax_a.set_xlabel('X (mm)')
ax_a.set_ylabel('Y (mm)')
ax_a.set_title('Trajectory', fontsize=18)
ax_a.legend(loc='lower left', frameon = False)
ax_a.set_xlim(-35, 115)
ax_a.set_ylim(-75, 20)
ax_a.text(-0.2, 1.15, 'a', transform=ax_a.transAxes, fontsize=24, fontweight='bold', va='top')

# panel b: Velocity profile 

ax_b = axes[1]
ax_b.set_facecolor(BG_COLOR)

# L-shape path
for i in range(len(L_PATH_X) - 1):
    ax_b.plot([L_PATH_X[i], L_PATH_X[i+1]], [L_PATH_Y[i], L_PATH_Y[i+1]],
              '-', color='gray', linewidth=5, alpha=0.3)

# Closed-loop velocity
cl_x = cl['x_mm'].values
cl_y = cl['y_mm'].values
cl_dx = np.diff(cl_x) / DT
cl_dy = np.diff(cl_y) / DT
cl_speed = np.sqrt(cl_dx**2 + cl_dy**2)
cl_speed_clipped = np.clip(cl_speed, 0, np.percentile(cl_speed, 95))

sc_cl = ax_b.scatter(cl_x[1:], cl_y[1:], c=cl_speed_clipped, cmap=cl_cmap, s=40, edgecolors='none', linewidths=0.1, zorder=4, vmin=vmin, vmax=vmax)

# Open-loop velocity
ol_x = ol['x_mm'].values
ol_y = ol['y_mm'].values
ol_dx = np.diff(ol_x) / (20 * DT)
ol_dy = np.diff(ol_y) / (20 * DT)
ol_speed = np.sqrt(ol_dx**2 + ol_dy**2)
ol_speed_clipped = np.clip(ol_speed, 0, vmax)

sc_ol = ax_b.scatter(ol_x[1:], ol_y[1:], c=ol_speed_clipped, cmap=ol_cmap, s=70, edgecolors='none', linewidths=0.1, zorder=5,
                      vmin=vmin, vmax=vmax, marker='s')

# Start and end markers
ax_b.plot(cl_x[0], cl_y[0], 'o', color='green', markersize=12, zorder=10)
ax_b.plot(cl_x[-1], cl_y[-1], '*', color='blue', markersize=14, zorder=10)
ax_b.plot(ol_x[-1], ol_y[-1], 'X', color='red', markersize=12, zorder=10)

ax_b.set_xlabel('X (mm)')
ax_b.set_ylabel('Y (mm)')
ax_b.set_title('Velocity profile', fontsize=18)
ax_b.set_xlim(-35, 115)
ax_b.set_ylim(-75, 20)

# Two colorbars side by side
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax_b)
cax1 = divider.append_axes("right", size="3%", pad=0.1)
cax2 = divider.append_axes("right", size="3%", pad=0.5)

cbar1 = fig.colorbar(sc_cl, cax=cax1)
cbar1.set_label('Closed-loop (mm/s)', fontsize=10)
cbar1.ax.tick_params(labelsize=10)

cbar2 = fig.colorbar(sc_ol, cax=cax2)
cbar2.set_label('Open-loop (mm/s)', fontsize=10)
cbar2.ax.tick_params(labelsize=10)

# Legend for marker shapes
legend_elements = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor='#0077B6', markersize=8, markeredgecolor='none', label='Closed-loop'),
    Line2D([0], [0], marker='s', color='none', markerfacecolor='#E63946', markersize=8, markeredgecolor='none', label='Open-loop'),
]
ax_b.legend(handles=legend_elements, loc='lower left', frameon=False, fontsize=16)

ax_b.text(-0.22, 1.15, 'b', transform=ax_b.transAxes, fontsize=24, fontweight='bold', va='top')

# panel c: Progress along path vs time

ax_c = axes[2]
ax_c.set_facecolor(BG_COLOR)

# Closed-loop progress
cl_time = np.arange(len(cl)) * DT
ax_c.plot(cl_time, cl['progress'].values, color=CL_COLOR, linewidth=2.5,
          label='Closed-loop', zorder=4)

# Open-loop progress: compute from position relative to waypoints
# Interpolate open-loop positions to all steps

ol_steps_all = np.arange(0, int(ol['step'].iloc[-1]) + 1)
ol_x_all = np.interp(ol_steps_all, ol['step'].values, ol['x_mm'].values)
ol_y_all = np.interp(ol_steps_all, ol['step'].values, ol['y_mm'].values)

# Compute progress for open-loop using L-shape waypoints
# Simple approximation: project onto the L-shape path segments

wp = np.array([[87, -64], [87, 3], [-26, 3]])  # L-shape waypoints
seg_lengths = [np.linalg.norm(wp[i+1] - wp[i]) for i in range(len(wp)-1)]
cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])

ol_progress = []
for xi, yi in zip(ol_x_all, ol_y_all):
    p = np.array([xi, yi])
    best_prog = 0
    for s in range(len(wp) - 1):
        a, b = wp[s], wp[s+1]
        ab = b - a
        ap = p - a
        t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
        proj_dist = cum_lengths[s] + t * seg_lengths[s]
        proj_point = a + t * ab
        dist_to_path = np.linalg.norm(p - proj_point)
        if dist_to_path < 60:  # within reasonable range
            if proj_dist > best_prog:
                best_prog = proj_dist
    ol_progress.append(best_prog)

ol_time = ol_steps_all * DT
ax_c.plot(ol_time, ol_progress, color=OL_COLOR, linewidth=2.5,
          label='Open-loop', zorder=3)

# Corner line
corner_progress = seg_lengths[0]  # ~67mm (first segment length)
ax_c.axhline(y=corner_progress, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
ax_c.text(0.82, 0.28, 'corner', transform=ax_c.transAxes, fontsize=12, color='gray', fontstyle='italic')


# Goal line
total_path = sum(seg_lengths)
ax_c.axhline(y=total_path, color='#2D6A4F', linestyle='--', linewidth=3, alpha=0.6)


ax_c.set_xlabel('Time (s)')
ax_c.set_ylabel('Progress along path (mm)')
ax_c.set_title('Progress', fontsize=18)
ax_c.set_xlim(0, max(cl_time[-1], ol_time[-1]) * 1.02)
ax_c.set_ylim(0, total_path * 1.15)
ax_c.legend(loc='lower right', frameon=False)
ax_c.text(-0.2, 1.15, 'c', transform=ax_c.transAxes,
          fontsize=24, fontweight='bold', va='top')

plt.savefig('fig_cl_vs_ol_1x3.png', bbox_inches='tight', dpi=600)

# Save as PDF for high-quality printing
plt.savefig('fig_cl_vs_ol_1x3.pdf', bbox_inches='tight', dpi=600)
plt.close()
