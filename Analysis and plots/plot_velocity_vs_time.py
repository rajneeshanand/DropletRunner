#!/usr/bin/env python3

#Droplet Velocity vs Time


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


LSHAPE_DIR = 'mbrl_actions_ep20'
ISHAPE_DIR = 'mbrl_actions_Ishape_ep20'
ARC_DIR = 'mbrl_actions_arc'
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ

#Smoothing window (moving average) to reduce noise
SMOOTH_WINDOW = 5

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 24,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})


COLORS = {
    'fast': '#1b4f72',    # dark blue
    'medium': '#7b2d8e',  # purple
    'slow': '#c0392b',    # red
}
LINEWIDTHS = {'fast': 3.0, 'medium': 2.0, 'slow': 1.6}
ALPHAS = {'fast': 1.0, 'medium': 0.85, 'slow': 0.7}


def load_successful_episodes(directory, min_progress=50, max_steps=590):
    eps = {}
    for f in sorted(os.listdir(directory)):
        if f.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, f))
            n = len(df)
            prog = df['progress'].iloc[-1]
            if prog > min_progress and n < max_steps:
                eps[f] = {'n': n, 'df': df}
    return sorted(eps.items(), key=lambda x: x[1]['n'])


def select_episodes(sorted_eps, fast_range=None, medium_range=None):
    if fast_range:
        candidates = [e for e in sorted_eps if fast_range[0] < e[1]['n'] < fast_range[1]]
        fast = candidates[0] if candidates else sorted_eps[0]
    else:
        fast = sorted_eps[0]

    if medium_range:
        candidates = [e for e in sorted_eps if medium_range[0] < e[1]['n'] < medium_range[1]]
        medium = candidates[0] if candidates else sorted_eps[len(sorted_eps)//2]
    else:
        medium = sorted_eps[len(sorted_eps)//2]

    slow = sorted_eps[-1]
    return fast, medium, slow


def compute_velocity(df, smooth=SMOOTH_WINDOW):
    x = df['x_mm'].values.astype(float)
    y = df['y_mm'].values.astype(float)
    dx = np.diff(x) / DT
    dy = np.diff(y) / DT
    speed = np.sqrt(dx**2 + dy**2)
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        speed = np.convolve(speed, kernel, mode='same')
    time = np.arange(len(speed)) * DT
    return time, speed


def plot_velocity_panel(ax, episodes, panel_label, title):
    labels = ['fast', 'medium', 'slow']

    for (fname, ep), label in zip(episodes, labels):
        df = ep['df']
        n = ep['n']
        t, v = compute_velocity(df)
        duration = n * DT
        lbl = f'{label.capitalize()} ({duration:.1f} s)'
        ax.plot(t, v, color=COLORS[label], linewidth=LINEWIDTHS[label],
                alpha=ALPHAS[label], label=lbl)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (mm/s)')
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=18, frameon=False, loc='upper right')
    ax.minorticks_on()
    ax.tick_params(which='minor', length=2)

    # Panel label
    ax.text(-0.15, 1.15, panel_label, transform=ax.transAxes,
            fontsize=40, fontweight='bold', va='top')



#Load episodes

ishape_sorted = load_successful_episodes(ISHAPE_DIR, min_progress=50)
lshape_sorted = load_successful_episodes(LSHAPE_DIR, min_progress=150)
arc_sorted = load_successful_episodes(ARC_DIR, min_progress=150)

print(f"I-shape: {len(ishape_sorted)} successful episodes")
print(f"L-shape: {len(lshape_sorted)} successful episodes")
print(f"Arc:     {len(arc_sorted)} successful episodes")

#Select fast/medium/slow
i_fast, i_medium, i_slow = select_episodes(ishape_sorted)
l_fast, l_medium, l_slow = select_episodes(lshape_sorted, fast_range=(30, 50), medium_range=(177, 180))
a_fast, a_medium, a_slow = select_episodes(arc_sorted, fast_range=(50, 60), medium_range=(100, 120))

print(f"\nI-shape  — Fast: {i_fast[1]['n']} steps, Medium: {i_medium[1]['n']} steps, Slow: {i_slow[1]['n']} steps")
print(f"L-shape  — Fast: {l_fast[1]['n']} steps, Medium: {l_medium[1]['n']} steps, Slow: {l_slow[1]['n']} steps")
print(f"Arc      — Fast: {a_fast[1]['n']} steps, Medium: {a_medium[1]['n']} steps, Slow: {a_slow[1]['n']} steps")

#Plot 1x3 grid
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

plot_velocity_panel(axes[0], [i_fast, i_medium, i_slow], 'a', 'I-shape')

plot_velocity_panel(axes[1], [l_fast, l_medium, l_slow], 'b', 'L-shape')

plot_velocity_panel(axes[2], [a_fast, a_medium, a_slow], 'c', 'Arc-shape')

plt.tight_layout(w_pad=1.0)
plt.savefig('fig_velocity_vs_time_1x3.png', bbox_inches='tight', dpi=600)

plt.savefig('fig_velocity_vs_time_1x3.pdf', bbox_inches='tight', dpi=600)