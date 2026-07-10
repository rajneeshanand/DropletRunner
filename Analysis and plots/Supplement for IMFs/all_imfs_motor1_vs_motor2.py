
#side-by-side Current IMFs: Motor 1 vs Motor 2 for visual comparison of all IMFs from both motor currents.


import pandas as pd
import numpy as np
from PyEMD import EMD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#Configuration and file paths
DATA_PATH = '../mbrl_actions_ep20/mbrl_actions_ep201122.csv'
OUTPUT_PATH = 'all_imfs_motor1_vs_motor2.png'
DT = 0.05
CORNER = (87.0, 3.0)
CORNER_RADIUS = 30.0
BG = '#0e1117'

df = pd.read_csv(DATA_PATH)
N = len(df)
t = np.arange(N) * DT

x = df['x_mm'].values.astype(float)
y = df['y_mm'].values.astype(float)
current1 = df['current1'].values.astype(float)
current2 = df['current2'].values.astype(float)


d2c = np.sqrt((x - CORNER[0])**2 + (y - CORNER[1])**2)          #corner zone
stuck_mask = d2c < CORNER_RADIUS
stuck_start = np.where(stuck_mask)[0][0]
stuck_end = np.where(stuck_mask)[0][-1]

# EMD on current1 (Motor#1)

emd1 = EMD()
cur1_imfs = emd1.emd(current1 - current1.mean())
n_cur1 = len(cur1_imfs)

# EMD on current2 (Motor#2)

emd2 = EMD()
cur2_imfs = emd2.emd(current2 - current2.mean())
n_cur2 = len(cur2_imfs)

def zc_freq(imf):
    zc = np.sum(np.diff(np.signbit(imf)))
    return (zc / 2) / (N * DT)

#classify IMFs based on significance and tests
# Motor 1 (current1):
#   0, 1, 6: NOISE (Wu-Huang)
#   4, 5: SIGNAL but FREE-ACTIVE (stuck energy test)
#   2: CANDIDATE (passed Wu-Huang + stuck energy, eliminated by freq match)
#   3: WINNER (all 5 tests)

cur1_labels = {
    0: ('NOISE — eliminated by Wu-Huang', '#ff4444'),
    1: ('NOISE — eliminated by Wu-Huang', '#ff4444'),
    2: ('CANDIDATE — passed Wu-Huang + stuck energy', '#00d4ff'),
    3: ('★ DEPINNING — winner of all 5 tests', '#00ff88'),
    4: ('FREE-ACTIVE — eliminated by stuck energy test', '#cc66ff'),
    5: ('FREE-ACTIVE — eliminated by stuck energy test', '#ffd700'),
    6: ('NOISE — eliminated by Wu-Huang', '#ff66cc'),
}

# Motor#2 (current2):
#   All except 3: NOISE (Wu-Huang alone eliminates everything)
#   3: ONLY SURVIVOR (95% significant)

cur2_labels = {}
for i in range(n_cur2):
    freq = zc_freq(cur2_imfs[i])
    if i == 3:
        cur2_labels[i] = ('★ DEPINNING — only Wu-Huang survivor (95%)', '#00ff88')
    else:
        cur2_labels[i] = (f'NOISE — eliminated by Wu-Huang', '#ff4444')


n_rows = max(n_cur1, n_cur2) + 1  # +1 for raw signal row
fig, axes = plt.subplots(n_rows, 2, figsize=(18, n_rows * 2.2), facecolor=BG,
                          sharex=True)

def style_ax(ax, border_color='#333', border_width=1):
    ax.set_facecolor(BG)
    ax.tick_params(colors='white', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(border_width)

def add_corner_shading(ax):
    ax.axvspan(stuck_start*DT, stuck_end*DT, color='#ffd700', alpha=0.07)
    ax.axvline(stuck_start*DT, color='#ffd700', linewidth=0.8,
               linestyle='--', alpha=0.4)
    ax.axvline(stuck_end*DT, color='#ffd700', linewidth=0.8,
               linestyle='--', alpha=0.4)

#Row 0: Raw signals
# Motor#1

ax = axes[0, 0]
style_ax(ax)
add_corner_shading(ax)
ax.plot(t, current1, color='white', linewidth=0.6)
ax.set_ylabel('mA', color='white', fontsize=16)
ax.set_title('MOTOR 1 (current1)', color='#4A90D9', fontsize=16, fontweight='bold')
ax.text(0.02, 0.82, 'Raw Signal', transform=ax.transAxes,
        color='white', fontsize=16, fontweight='bold')

# Motor#2

ax = axes[0, 1]
style_ax(ax)
add_corner_shading(ax)
ax.plot(t, current2, color='white', linewidth=0.6)
ax.set_ylabel('mA', color='white', fontsize=16)
ax.set_title('MOTOR 2 (current2)', color='#D4654A', fontsize=16, fontweight='bold')
ax.text(0.02, 0.82, 'Raw Signal', transform=ax.transAxes,
        color='white', fontsize=16, fontweight='bold')

#Motor#1 IMFs

for i in range(n_cur1):
    row = i + 1
    ax = axes[row, 0]

    freq = zc_freq(cur1_imfs[i])
    energy = np.var(cur1_imfs[i])
    label_text, color = cur1_labels.get(i, ('', '#888'))

    is_winner = (i == 3)
    is_candidate = (i in [2, 3])
    border = '#00ff88' if is_winner else ('#00d4ff' if is_candidate else '#333')
    bw = 2.5 if is_winner else (1.5 if is_candidate else 1)

    style_ax(ax, border_color=border, border_width=bw)
    add_corner_shading(ax)
    ax.axhline(0, color='#444', linewidth=0.3)

    lw = 1.2 if is_winner else 0.8
    ax.plot(t, cur1_imfs[i], color=color, linewidth=lw)

    ax.text(0.01, 0.82, f'IMF {i}  |  {freq:.3f} Hz  |  E={energy:.0f}',
            transform=ax.transAxes, color=color, fontsize=14, fontweight='bold')

    status_color = '#00ff88' if is_winner else (color if is_candidate else '#ff4444')
    ax.text(0.99, 0.82, label_text, transform=ax.transAxes,
            color=status_color, fontsize=14, ha='right',
            fontweight='bold' if is_winner else 'normal')

    ax.set_ylabel('mA', color='white', fontsize=16)

#Motor#2 IMFs (right column)
for i in range(n_cur2):
    row = i + 1
    ax = axes[row, 1]

    freq = zc_freq(cur2_imfs[i])
    energy = np.var(cur2_imfs[i])
    label_text, color = cur2_labels.get(i, ('', '#888'))

    is_winner = (i == 3)
    border = '#00ff88' if is_winner else '#333'
    bw = 2.5 if is_winner else 1

    style_ax(ax, border_color=border, border_width=bw)
    add_corner_shading(ax)
    ax.axhline(0, color='#444', linewidth=0.3)

    lw = 1.2 if is_winner else 0.8
    ax.plot(t, cur2_imfs[i], color=color, linewidth=lw)

    ax.text(0.01, 0.82, f'IMF {i}  |  {freq:.3f} Hz  |  E={energy:.0f}',
            transform=ax.transAxes, color=color, fontsize=14, fontweight='bold')

    status_color = '#00ff88' if is_winner else '#ff4444'
    ax.text(0.99, 0.82, label_text, transform=ax.transAxes,
            color=status_color, fontsize=14, ha='right',
            fontweight='bold' if is_winner else 'normal')

    ax.set_ylabel('mA', color='white', fontsize=16)

# Hide extra rows
for row in range(n_cur1 + 1, n_rows):
    axes[row, 0].set_visible(False)
for row in range(n_cur2 + 1, n_rows):
    axes[row, 1].set_visible(False)

# Bottom x-labels
axes[n_cur1, 0].set_xlabel('Time (s)', color='white', fontsize=16)
axes[n_cur2, 1].set_xlabel('Time (s)', color='white', fontsize=16)

# Footer
fig.text(0.5, -0.01,
         'Motor 1: IMF 3 (0.653 Hz, 99% significant) ←→ Motor 2: IMF 3 (0.582 Hz, 95% significant)\n'
         'Both motors independently identify IMF 3 as the depinning oscillation  |  '
         'Yellow shading = corner zone',
         ha='center', color='#00ff88', fontsize=11, fontweight='bold',
         linespacing=1.5)

plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"\nSaved: {OUTPUT_PATH}")
