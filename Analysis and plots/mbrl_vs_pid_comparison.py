#!/usr/bin/env python3

# DropletRunner: MBRL vs PID Baseline 

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from pathlib import Path
import glob
import os



#dirctory path for data files (CSV episodes)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR

MBRL_I_PATTERN = str(DATA_ROOT / "mbrl_actions_Ishape_ep20" / "*.csv")  # MBRL, I-shape
MBRL_L_PATTERN = str(DATA_ROOT / "mbrl_actions_ep20" / "*.csv")         # MBRL, L-shape
PID_I_PATTERN  = str(DATA_ROOT / "pid_actions_Ishape" / "*.csv")        # PID, I-shape
PID_L_PATTERN  = str(DATA_ROOT / "pid_actions_Lshape" / "*.csv")        # PID, L-shape
MBRL_A_PATTERN = str(DATA_ROOT / "mbrl_actions_arc" / "*.csv")          # MBRL, Arc-shape
PID_A_PATTERN  = str(DATA_ROOT / "pid_actions_ARC" / "*.csv")            # PID, Arc-shape



MBRL_COLOR = "#0077B6"
MBRL_LIGHT = "#90E0EF"

PID_COLOR = "#E63946"
PID_LIGHT = "#FFCCD5"

BG_COLOR = "white"
GRID_COLOR = "#E5E5E5"

I_PATH_LENGTH = 113.0     # mm
L_PATH_LENGTH = 173.8     # mm
A_PATH_LENGTH = 208.3     # mm
DT = 0.05                 # 20 Hz control loop

def smooth(data, window=11):
    #Simple moving-average smoother.
    data = np.asarray(data, dtype=float)

    if data.ndim == 0:
        return data
    if len(data) == 0:
        return data
    if len(data) < window:
        return data.copy()

    kernel = np.ones(window) / window
    pad = window // 2

    padded = np.concatenate([
        np.full(pad, data[0]),
        data,
        np.full(pad, data[-1])
    ])
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[:len(data)]


def load_episodes(pattern, name):
    
    #Load all CSV episodes from a glob pattern.
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        print(f"WARNING: No files found for {name}")
        print(f"Pattern checked: {pattern}")
        return []

    episodes = []

    for f in files:
        df = pd.read_csv(f)

        #Time from row index
        df["time_s"] = np.arange(len(df)) * DT

        #Check required columns
        if "progress" not in df.columns:
            raise ValueError(
                f"'progress' column not found in file:\n{f}\n"
                f"Available columns: {list(df.columns)}"
            )

        episodes.append(df)

    print(f"{name}: loaded {len(files)} episodes")
    return episodes


def compute_envelope(episodes, max_time, name="dataset"):           #Interpolate all episodes onto a common time grid.
    n_points = int(max_time / DT) + 1
    time_grid = np.linspace(0, max_time, n_points)

    if len(episodes) == 0:
        raise ValueError(
            f"\nNo episodes found for {name}.\n"
            f"Please check whether the corresponding folder exists and contains CSV files.\n"
        )

    interpolated = []

    for ep in episodes:
        t = ep["time_s"].values
        p = ep["progress"].values

        interp_p = np.interp(
            time_grid,
            t,
            p,
            left=0.0,
            right=float(p[-1])
        )

        interpolated.append(interp_p)
    interpolated = np.array(interpolated)

    mean = np.mean(interpolated, axis=0)
    std = np.std(interpolated, axis=0)
    return time_grid, mean, std, interpolated


def episode_stats(episodes, path_length, name="dataset"):               #Compute per-episode success and progress statistics.
    if len(episodes) == 0:
        raise ValueError(f"No episodes available for {name}")

    results = []

    for ep in episodes:
        max_prog = float(ep["progress"].max())
        n_steps = len(ep)
        success = max_prog >= path_length * 0.90

        results.append({
            "max_progress": max_prog,
            "steps": n_steps,
            "time_s": n_steps * DT,
            "success": success,
            "progress_pct": 100.0 * max_prog / path_length
        })

    return pd.DataFrame(results)


def action_metrics(episodes, name="dataset"):            #Compute action-based metrics, e.g. sign-change rate.
    if len(episodes) == 0:
        raise ValueError(f"No episodes available for {name}")

    rows = []
    for ep in episodes:

        if "action_norm1" not in ep.columns or "action_norm2" not in ep.columns:
            raise ValueError(
                f"Columns 'action_norm1' and/or 'action_norm2' are missing in {name}.\n"
                f"Available columns: {list(ep.columns)}"
            )

        a1 = ep["action_norm1"].values
        a2 = ep["action_norm2"].values

        duration_s = max(len(ep) * DT, DT)

        sc1 = np.sum(np.diff(np.sign(a1)) != 0) / duration_s
        sc2 = np.sum(np.diff(np.sign(a2)) != 0) / duration_s

        rows.append({
            "sign_change_rate": 0.5 * (sc1 + sc2),
            "max_progress": float(ep["progress"].max())
        })

    return pd.DataFrame(rows)

# Load data

print("\nLoading data...\n")

mbrl_I = load_episodes(MBRL_I_PATTERN, "MBRL I-shape")
mbrl_L = load_episodes(MBRL_L_PATTERN, "MBRL L-shape")
pid_I  = load_episodes(PID_I_PATTERN,  "PID I-shape")
pid_L  = load_episodes(PID_L_PATTERN,  "PID L-shape")
mbrl_A = load_episodes(MBRL_A_PATTERN, "MBRL Arc")
pid_A  = load_episodes(PID_A_PATTERN,  "PID Arc")

print("\nDataset summary:")
print(f"MBRL I-shape episodes: {len(mbrl_I)}")
print(f"MBRL L-shape episodes: {len(mbrl_L)}")
print(f"PID I-shape episodes:  {len(pid_I)}")
print(f"PID L-shape episodes:  {len(pid_L)}")

# MBRL episodes 
print("\n=== MBRL I-shape episodes ===")
for i, ep in enumerate(mbrl_I):
    print(f"  [{i}] steps={len(ep)}, time={len(ep)*DT:.1f}s, progress={ep['progress'].max():.1f}mm")

print("\n=== MBRL L-shape episodes ===")
for i, ep in enumerate(mbrl_L):
    print(f"  [{i}] steps={len(ep)}, time={len(ep)*DT:.1f}s, progress={ep['progress'].max():.1f}mm")

#  MBRL selection: [fast, median, slow]

mbrl_I_sel = [mbrl_I[3], mbrl_I[5], mbrl_I[13]]
mbrl_L_sel = [mbrl_L[1], mbrl_L[5], mbrl_L[12]]

# PID episodes 
print("\n=== PID I-shape episodes ===")
for i, ep in enumerate(pid_I):
    print(f"  [{i}] steps={len(ep)}, time={len(ep)*DT:.1f}s, progress={ep['progress'].max():.1f}mm")

print("\n=== PID L-shape episodes ===")
for i, ep in enumerate(pid_L):
    print(f"  [{i}] steps={len(ep)}, time={len(ep)*DT:.1f}s, progress={ep['progress'].max():.1f}mm")

#  PID selection: [fast, median, slow]
pid_I_sel = [pid_I[7], pid_I[3], pid_I[9]]
pid_L_sel = [pid_L[1], pid_L[3], pid_L[9]]


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.3,
    "xtick.direction": "in",
    "ytick.direction": "in"
})


fig = plt.figure(figsize=(10.5, 9.5))

gs = GridSpec(
    3, 3,
    figure=fig,
    height_ratios=[0.5, 1, 1.6],
    hspace=0.40,
    wspace=0.30,
    left=0.07,
    right=0.98,
    top=0.97,
    bottom=0.14
)

from matplotlib.image import imread

ax_img_I = fig.add_subplot(gs[0, 0])
img_I = imread("shapeImages/I.png")
for spine in ax_img_I.spines.values():
    spine.set_visible(False)
ax_img_I.imshow(img_I)
ax_img_I.set_xticks([])
ax_img_I.set_yticks([])

ax_img_I.text(
    -1.2, 1.10, "a",
    transform=ax_img_I.transAxes,
    fontsize=18,
    fontweight="bold",
    va="top",
    ha="right"
)

ax_img_L = fig.add_subplot(gs[0, 1])
img_L = imread("shapeImages/L.png")
for spine in ax_img_L.spines.values():
    spine.set_visible(False)
ax_img_L.imshow(img_L)
ax_img_L.set_xticks([])
ax_img_L.set_yticks([])

ax_img_L.text(
    -1.2, 1.10, "b",
    transform=ax_img_L.transAxes,
    fontsize=18,
    fontweight="bold",
    va="top",
    ha="right"
)


ax_img_A = fig.add_subplot(gs[0, 2])
img_A = imread("shapeImages/ARC.png")
for spine in ax_img_A.spines.values():
    spine.set_visible(False)
ax_img_A.imshow(img_A)
ax_img_A.set_xticks([])
ax_img_A.set_yticks([])
ax_img_A.text(-1.2, 1.10, "c", transform=ax_img_A.transAxes,
              fontsize=18, fontweight="bold", va="top", ha="left")



# PANEL (a): I-SHAPE PROGRESS VS TIME


ax_a = fig.add_subplot(gs[1, 0])
ax_a.set_facecolor(BG_COLOR)
ax_a.grid(True, alpha=0.30, color=GRID_COLOR, linewidth=0.4)

max_time_I = 12.0

t_I, mean_mbrl_I, std_mbrl_I, all_mbrl_I = compute_envelope(
    mbrl_I, max_time_I, "MBRL I-shape"
)

_, mean_pid_I, std_pid_I, all_pid_I = compute_envelope(
    pid_I, max_time_I, "PID I-shape"
)

# Individual trajectories
for i in range(all_mbrl_I.shape[0]):
    ax_a.plot(
        t_I, all_mbrl_I[i],
        color=MBRL_COLOR,
        alpha=0.045,
        linewidth=0.35
    )

for i in range(all_pid_I.shape[0]):
    ax_a.plot(
        t_I, all_pid_I[i],
        color=PID_COLOR,
        alpha=0.06,
        linewidth=0.35
    )

# Standard-deviation bands
ax_a.fill_between(
    t_I,
    np.clip(mean_mbrl_I - std_mbrl_I, 0, None),
    mean_mbrl_I + std_mbrl_I,
    color=MBRL_LIGHT,
    alpha=0.40,
    edgecolor="none"
)

ax_a.fill_between(
    t_I,
    np.clip(mean_pid_I - std_pid_I, 0, None),
    mean_pid_I + std_pid_I,
    color=PID_LIGHT,
    alpha=0.38,
    edgecolor="none"
)

# Mean curves
ax_a.plot(
    t_I,
    smooth(mean_mbrl_I, 15),
    color=MBRL_COLOR,
    linewidth=2.8,
    label="MBRL",
    zorder=5
)

ax_a.plot(
    t_I,
    smooth(mean_pid_I, 15),
    color=PID_COLOR,
    linewidth=2.8,
    label="PID",
    zorder=5
)

# Goal line
ax_a.axhline(
    y=I_PATH_LENGTH,
    color="#2D6A4F",
    linestyle="--",
    linewidth=0.8,
    alpha=0.75
)

ax_a.set_xlabel("Time (s)")
ax_a.set_ylabel("Progress along path (mm)")
ax_a.set_xlim(0, max_time_I)
ax_a.set_ylim(-5, I_PATH_LENGTH * 1.15)

ax_a.legend(
    loc="lower right",
    bbox_to_anchor=(0.98, 0.12),
    frameon=False,
    fancybox=False,
    edgecolor="#CCCCCC",
    facecolor="white",
    framealpha=0.95,
    borderpad=0.4
)

ax_a.set_title("I-shape (straight, 113 mm)", pad=8)



# PANEL (b): L-SHAPE PROGRESS VS TIME


ax_b = fig.add_subplot(gs[1, 1])
ax_b.set_facecolor(BG_COLOR)
ax_b.grid(True, alpha=0.30, color=GRID_COLOR, linewidth=0.4)

max_time_L = 30.0

t_L, mean_mbrl_L, std_mbrl_L, all_mbrl_L = compute_envelope(
    mbrl_L, max_time_L, "MBRL L-shape"
)

_, mean_pid_L, std_pid_L, all_pid_L = compute_envelope(
    pid_L, max_time_L, "PID L-shape"
)

# Individual trajectories
for i in range(all_mbrl_L.shape[0]):
    ax_b.plot(
        t_L, all_mbrl_L[i],
        color=MBRL_COLOR,
        alpha=0.045,
        linewidth=0.35
    )

for i in range(all_pid_L.shape[0]):
    ax_b.plot(
        t_L, all_pid_L[i],
        color=PID_COLOR,
        alpha=0.06,
        linewidth=0.35
    )

# Standard-deviation bands
ax_b.fill_between(
    t_L,
    np.clip(mean_mbrl_L - std_mbrl_L, 0, None),
    mean_mbrl_L + std_mbrl_L,
    color=MBRL_LIGHT,
    alpha=0.40,
    edgecolor="none"
)

ax_b.fill_between(
    t_L,
    np.clip(mean_pid_L - std_pid_L, 0, None),
    mean_pid_L + std_pid_L,
    color=PID_LIGHT,
    alpha=0.38,
    edgecolor="none"
)

# Mean curves
ax_b.plot(
    t_L,
    smooth(mean_mbrl_L, 31),
    color=MBRL_COLOR,
    linewidth=2.8,
    label="MBRL",
    zorder=5
)

ax_b.plot(
    t_L,
    smooth(mean_pid_L, 31),
    color=PID_COLOR,
    linewidth=2.8,
    label="PID",
    zorder=5
)

# Goal line
ax_b.axhline(
    y=L_PATH_LENGTH,
    color="#2D6A4F",
    linestyle="--",
    linewidth=0.8,
    alpha=0.75
)

ax_b.set_xlabel("Time (s)")
ax_b.set_ylabel("Progress along path (mm)")
ax_b.set_xlim(0, max_time_L)
ax_b.set_ylim(-5, L_PATH_LENGTH * 1.15)

ax_b.legend(
    loc="lower right",
    bbox_to_anchor=(0.98, 0.12),
    frameon=False,
    fancybox=False,
    edgecolor="#CCCCCC",
    facecolor="white",
    framealpha=0.95,
    borderpad=0.4
)
ax_b.set_title("L-shape (turn, 173.8 mm)", pad=8)



# PANEL (c): ARC PROGRESS VS TIME


ax_arc = fig.add_subplot(gs[1, 2])
ax_arc.set_facecolor(BG_COLOR)
ax_arc.grid(True, alpha=0.30, color=GRID_COLOR, linewidth=0.4)

max_time_A = 30.0

t_A, mean_mbrl_A, std_mbrl_A, all_mbrl_A = compute_envelope(
    mbrl_A, max_time_A, "MBRL Arc"
)
_, mean_pid_A, std_pid_A, all_pid_A = compute_envelope(
    pid_A, max_time_A, "PID Arc"
)

for i in range(all_mbrl_A.shape[0]):
    ax_arc.plot(t_A, all_mbrl_A[i], color=MBRL_COLOR, alpha=0.045, linewidth=0.35)
for i in range(all_pid_A.shape[0]):
    ax_arc.plot(t_A, all_pid_A[i], color=PID_COLOR, alpha=0.06, linewidth=0.35)

ax_arc.fill_between(t_A, np.clip(mean_mbrl_A - std_mbrl_A, 0, None),
                    mean_mbrl_A + std_mbrl_A,
                    color=MBRL_LIGHT, alpha=0.40, edgecolor="none")
ax_arc.fill_between(t_A, np.clip(mean_pid_A - std_pid_A, 0, None),
                    mean_pid_A + std_pid_A,
                    color=PID_LIGHT, alpha=0.38, edgecolor="none")

ax_arc.plot(t_A, smooth(mean_mbrl_A, 31), color=MBRL_COLOR,
            linewidth=2.8, label="MBRL", zorder=5)
ax_arc.plot(t_A, smooth(mean_pid_A, 31), color=PID_COLOR,
            linewidth=2.8, label="PID", zorder=5)

ax_arc.axhline(y=A_PATH_LENGTH, color="#2D6A4F", linestyle="--",
               linewidth=0.8, alpha=0.75)

ax_arc.set_xlabel("Time (s)")
ax_arc.set_ylabel("Progress along path (mm)")
ax_arc.set_xlim(0, max_time_A)
ax_arc.set_ylim(-5, A_PATH_LENGTH * 1.15)
ax_arc.legend(loc="lower right", bbox_to_anchor=(1.0, 0.15),
              frameon=False, fancybox=False, edgecolor="#CCCCCC",
              facecolor="white", framealpha=0.95, borderpad=0.4)
ax_arc.set_title("Arc (curve, 208.3 mm)", pad=8)



# PANEL (d): SUCCESS RATE

ax_d = fig.add_subplot(gs[2, 0])
_p = ax_d.get_position()
ax_d.set_position([_p.x0, _p.y0 + _p.height * 0.10, _p.width, _p.height * 0.90])
ax_d.set_facecolor(BG_COLOR)

s_mbrl_I = episode_stats(mbrl_I, I_PATH_LENGTH, "MBRL I-shape")
s_pid_I  = episode_stats(pid_I,  I_PATH_LENGTH, "PID I-shape")
s_mbrl_L = episode_stats(mbrl_L, L_PATH_LENGTH, "MBRL L-shape")
s_pid_L  = episode_stats(pid_L,  L_PATH_LENGTH, "PID L-shape")
s_mbrl_A = episode_stats(mbrl_A, A_PATH_LENGTH, "MBRL Arc")
s_pid_A  = episode_stats(pid_A,  A_PATH_LENGTH, "PID Arc")

success_rates = [
    s_mbrl_I["success"].mean() * 100,
    s_pid_I["success"].mean() * 100,
    s_mbrl_L["success"].mean() * 100,
    s_pid_L["success"].mean() * 100,
    s_mbrl_A["success"].mean() * 100,
    s_pid_A["success"].mean() * 100,
]
colors_bar = [MBRL_COLOR, PID_COLOR, MBRL_COLOR, PID_COLOR, MBRL_COLOR, PID_COLOR]
x_pos = np.array([0, 1, 2.5, 3.5, 5.0, 6.0])
bars = ax_d.bar(
    x_pos, success_rates, width=0.70,
    color=colors_bar, alpha=0.85,
    edgecolor="white", linewidth=0.8, zorder=3
)

all_stats = [s_mbrl_I, s_pid_I, s_mbrl_L, s_pid_L, s_mbrl_A, s_pid_A]

# Individual episode dots: success vs failure
np.random.seed(42)

for i, stats in enumerate(all_stats):

    jitter = np.random.uniform(-0.15, 0.15, len(stats))
    progress_vals = stats["progress_pct"].values
    success_mask = progress_vals >= 90.0
    fail_mask = ~success_mask

    # Failed episodes: open white circles with dark edge
    ax_d.scatter(
        x_pos[i] + jitter[fail_mask],
        progress_vals[fail_mask],
        s=34,
        facecolors="white",
        edgecolors="#333333",
        linewidth=1.1,
        zorder=6,
        alpha=1.0,
        marker="o"
    )

    # Successful episodes: filled green circles
    ax_d.scatter(
        x_pos[i] + jitter[success_mask],
        progress_vals[success_mask],
        s=38,
        facecolors="#2D6A4F",
        edgecolors="white",
        linewidth=0.8,
        zorder=7,
        alpha=0.95,
        marker="o"
    )

# Percentage labels + counts
for bar, rate, stats in zip(bars, success_rates, all_stats):

    n_success = int(stats["success"].sum())
    n_total = len(stats)

    x_center = bar.get_x() + bar.get_width() / 2
    y_top = bar.get_height()

ax_d.set_xticks(x_pos)
ax_d.set_xticklabels([
    "I-shape\nMBRL", "I-shape\nPID",
    "L-shape\nMBRL", "L-shape\nPID",
    "Arc\nMBRL", "Arc\nPID"
], fontsize=7.5)
ax_d.axvline(x=4.25, color="#CCCCCC", linestyle="-", linewidth=0.5, alpha=0.5)

# Thumbnails placed as figure-level axes just below panel d
# so panel d's bar chart uses the full cell height (same as panels e and f)
_pos = ax_d.get_position()  # [x0, y0, width, height] in figure fraction
_thumb_h   = 0.06           # thumbnail strip height in figure fraction
_thumb_gap = 0.025           # gap between bar chart bottom and thumbnails
_thumb_w   = _pos.width / 3 - 0.004

#images below panel d, placed as figure-level axes

for _i, _img in enumerate([img_I, img_L, img_A]):
    _tx = _pos.x0 + _i * (_pos.width / 3)
    _ty = _pos.y0 - _thumb_gap - _thumb_h
    _tax = fig.add_axes([_tx, _ty, _thumb_w, _thumb_h])
    _tax.imshow(_img)
    _tax.set_xticks([])
    _tax.set_yticks([])
    for _sp in _tax.spines.values():
        _sp.set_visible(False)


ax_d.set_ylabel("Success rate / Episode max progress (%)")
ax_d.set_ylim(0, 115)

ax_d.grid(
    True,
    axis="y",
    alpha=0.25,
    color=GRID_COLOR,
    linewidth=0.4
)

ax_d.axvline(
    x=1.75,
    color="#CCCCCC",
    linestyle="-",
    linewidth=0.5,
    alpha=0.5
)

ax_d.set_title("Navigation success rate", pad=8)

ax_d.text(
    -0.18, 1.10, "d",
    transform=ax_d.transAxes,
    fontsize=18,
    fontweight="bold",
    va="top"
)



# PANEL (e): 2D TRAJECTORY COMPARISON


# Create two stacked sub-axes inside the gs[2,1] cell
gs_d = gs[2, 1].subgridspec(2, 1, hspace=0.55)
ax_d1 = fig.add_subplot(gs_d[0])  # I-shape
ax_d2 = fig.add_subplot(gs_d[1])  # L-shape

# Load waypoints
import json

# Waypoints (hardcoded, same as velocity profile plot)
wp_I = np.array([[87.5, -2.0], [-21.5, -2.0]])
wp_L = np.array([[87, -64], [87, 3], [-26, 3]])

def pick_representatives(episodes, name=""):                            #Pick best, median, worst episodes by max progress.
    progs = [float(ep["progress"].max()) for ep in episodes]
    steps = [len(ep) for ep in episodes]
    indices = np.argsort(progs)
    best   = indices[-1]
    worst  = indices[0]
    median = indices[len(indices) // 2]
    
    for label, idx in [("Best", best), ("Median", median), ("Worst", worst)]:
        print(f"  {name} {label}: episode {idx}, "
              f"{steps[idx]} steps ({steps[idx]*DT:.1f}s), "
              f"progress={progs[idx]:.1f}mm")
    
    return [episodes[best], episodes[median], episodes[worst]]

def plot_trajectories(ax, waypoints, mbrl_reps, pid_reps, title):
    ax.set_facecolor(BG_COLOR)

    # Path centerline
    ax.plot(
        waypoints[:, 0], waypoints[:, 1], color="#AAAAAA", linewidth=3.5, linestyle="-", zorder=1, solid_capstyle="round")

    # Path deviation band
    ax.plot(
        waypoints[:, 0], waypoints[:, 1],
        color="#E8E8E8", linewidth=12, linestyle="-",
        zorder=0, solid_capstyle="round"
    )

    #pid_shades = ["#C1121F", "#E63946", "#FFAAB0"]
    pid_shades = ["#E31A1C", "#FF7F00", "#CAB200"]
    pid_labels_style = ["fast", "median", "slow"]
    for ep, shade, style in zip(pid_reps, pid_shades, pid_labels_style):
        n = len(ep)
        ax.plot(
            ep["x_mm"].values, ep["y_mm"].values,
            color=shade, alpha=0.85, linewidth=1.2,
            zorder=3, linestyle="--",
            #label=f"PID {style} ({n} steps, {n*DT:.1f}s)"
            label=f"PID {style} ({n*DT:.1f}s)"
        )

    # MBRL trajectories 
    #mbrl_shades = ["#023E8A", "#0077B6", "#90E0EF"]
    mbrl_shades = ["#1F78B4", "#33A02C", "#B07CC6"]
    mbrl_labels_style = ["fast", "median", "slow"]
    for ep, shade, style in zip(mbrl_reps, mbrl_shades, mbrl_labels_style):
        n = len(ep)
        ax.plot(
            ep["x_mm"].values, ep["y_mm"].values,
            color=shade, alpha=0.85, linewidth=1.5,
            zorder=4,
            #label=f"MBRL {style} ({n} steps, {n*DT:.1f}s)"
            label=f"MBRL {style} ({n*DT:.1f}s)"
        )

    # Start and goal markers
    ax.scatter(waypoints[0, 0], waypoints[0, 1], s=150, c="#2D6A4F", edgecolor="#2D6A4F", linewidth=0.1, zorder=6, marker="o")
    ax.scatter(waypoints[-1, 0], waypoints[-1, 1], s=250, c="#E63946", edgecolor="#E63946", linewidth=0.1, zorder=6, marker="*")

    ax.set_aspect("auto")
    ax.set_title(title, fontsize=11, pad=4)
    ax.set_xlabel("x (mm)", fontsize=12)
    ax.set_ylabel("y (mm)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.15, color=GRID_COLOR, linewidth=0.3)
    ax.set_xlim(-35, 110)


# Plot both
plot_trajectories(ax_d1, wp_I, mbrl_I_sel, pid_I_sel, "I-shape (fast / median / slow)")
plot_trajectories(ax_d2, wp_L, mbrl_L_sel, pid_L_sel, "L-shape (fast / median / slow)")

# I-shape legend: MBRL upper-left, PID lower-left
handles_i, labels_i = ax_d1.get_legend_handles_labels()
mbrl_handles = [h for h, l in zip(handles_i, labels_i) if l.startswith("MBRL")]
mbrl_labels  = [l for l in labels_i if l.startswith("MBRL")]
pid_handles  = [h for h, l in zip(handles_i, labels_i) if l.startswith("PID")]
pid_labels   = [l for l in labels_i if l.startswith("PID")]

leg1 = ax_d1.legend(
    mbrl_handles, mbrl_labels,
    loc="lower right", fontsize=8.5,
    frameon=False, fancybox=False,
    edgecolor="#CCCCCC", facecolor="white",
    framealpha=0.92, borderpad=0.03,
    handlelength=1.5
)
ax_d1.add_artist(leg1)
ax_d1.legend(
    pid_handles, pid_labels,
    loc="lower left", fontsize=8.5,
    frameon=False, fancybox=False,
    edgecolor="#CCCCCC", facecolor="white",
    framealpha=0.92, borderpad=0.03,
    handlelength=1.5
)

# L-shape legend: all 6 in lower-left
handles_l, labels_l = ax_d2.get_legend_handles_labels()
ax_d2.legend(
    handles_l, labels_l,
    loc="lower left", fontsize=8.5, ncol=1,
    frameon=False, fancybox=False,
    edgecolor="#CCCCCC", facecolor="white",
    framealpha=0.92, borderpad=0.03,
    handlelength=1.5, columnspacing=0.6
)

# Panel label
ax_d1.text(
    -0.18, 1.25, "e",
    transform=ax_d1.transAxes,
    fontsize=18,
    fontweight="bold",
    va="top"
)



# PANEL (f): ABLATION STUDY



ax_f = fig.add_subplot(gs[2, 2])
ax_f.set_facecolor(BG_COLOR)

# Data
configs = ['50K steps\n50 ep', '50K steps\n100 ep', '150K steps\n100 ep', '450K steps\n100 ep']
x_abl = np.arange(len(configs)) * 1.1
bar_width = 0.2
offset = 0.2

I_success = [100, None, 100, 95]
L_success = [30, 95, 95, 90]
A_success = [40, 100, 95, 90]

I_ABL_COLOR = '#264653'
L_ABL_COLOR = '#580F41'
A_ABL_COLOR = '#E76F51'

# I-shape bars (left)
I_vals, I_x = [], []
for i, val in enumerate(I_success):
    if val is not None:
        I_vals.append(val)
        I_x.append(x_abl[i] - offset)
ax_f.bar(I_x, I_vals, width=bar_width, color=I_ABL_COLOR, alpha=0.88,
         edgecolor='none', zorder=3, label='I')

# L-shape bars (center)
ax_f.bar(x_abl, L_success, width=bar_width, color=L_ABL_COLOR, alpha=0.88,
         edgecolor='none', zorder=3, label='L')

# Arc bars (right)
ax_f.bar(x_abl + offset, A_success, width=bar_width, color=A_ABL_COLOR, alpha=0.88,
         edgecolor='none', zorder=3, label='Arc')


# L-shape improvement arrow (50ep → 100ep) — text ABOVE arrow
ax_f.annotate('', xy=(x_abl[1], 93), xytext=(x_abl[0], 29),
              arrowprops=dict(arrowstyle='->', color='black', lw=2.0))
ax_f.text((x_abl[0] + x_abl[1]) / 2 - 0.1, 55, '30→95%',
          ha='center', fontsize=10, color='black', fontweight='bold',
          rotation=67, va='bottom')

# Arc improvement arrow (50ep → 100ep) — text BELOW arrow
ax_f.annotate('', xy=(x_abl[1] + offset, 98), xytext=(x_abl[0] + 0.25, 39),
              arrowprops=dict(arrowstyle='->', color='black', lw=2.0))
ax_f.text((x_abl[0] + x_abl[1]) / 2 + offset - 0.1, 58, '40→100%',
          ha='center', fontsize=10, color='black', fontweight='bold',
          rotation=67, va='top')


# Overtraining annotation (450K) — three arrows merging to one label
merge_x = x_abl[3] + 0.55
merge_y = 60

# Arrow from I-shape bar (95%)
ax_f.annotate('', xy=(x_abl[3] - offset, 95),
              xytext=(merge_x, merge_y),
              arrowprops=dict(arrowstyle='->', color='black', lw=2.0))
# Arrow from L-shape bar (90%)
ax_f.annotate('', xy=(x_abl[3], 90),
              xytext=(merge_x, merge_y),
              arrowprops=dict(arrowstyle='->', color='black', lw=2.0))
# Arrow from Arc bar (90%)
ax_f.annotate('', xy=(x_abl[3] + offset, 90),
              xytext=(merge_x, merge_y),
              arrowprops=dict(arrowstyle='->', color='black', lw=2.0))
# Shared label with padding
ax_f.text(merge_x+0.1, 30, 'Overtraining',
          ha='center', fontsize=11, color='Red', fontweight='bold', rotation=90)

ax_f.set_xticks(x_abl)
ax_f.set_xticklabels(configs, fontsize=12)
ax_f.set_ylabel('Success rate (%)', fontsize=12)
ax_f.set_ylim(0, 112)
ax_f.set_xlim(-0.5, len(configs) + 0.1)
ax_f.tick_params(axis='y', labelsize=12)
ax_f.tick_params(axis='x', labelsize=10)
ax_f.grid(True, axis='y', alpha=0.25, color=GRID_COLOR, linewidth=0.4)
ax_f.set_title("Ablation: Training configuration", pad=8)

ax_f.legend(loc='upper right', frameon=False, fancybox=False,
            fontsize=12, ncol=3)

ax_f.text(0.95, -0.15, 'K = 1,000×', transform=ax_f.transAxes,
          fontsize=8, ha='right', va='bottom', color='gray')

ax_f.text(-0.18, 1.1, "f", transform=ax_f.transAxes,
          fontsize=18, fontweight="bold", va="top")




out_png = DATA_ROOT / "fig_mbrl_vs_pid_comparison.png"
fig.savefig(out_png, format="png", bbox_inches="tight", pad_inches=0.10, dpi=600)

# PDF format 
out_pdf = DATA_ROOT / "fig_mbrl_vs_pid_comparison.pdf"
fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.10)

plt.close(fig)
