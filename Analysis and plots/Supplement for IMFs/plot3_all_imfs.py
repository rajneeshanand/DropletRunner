
#Plot 3: Side-by-Side Current and Position IMFs
#PURPOSE: Visual comparison of all IMFs from motor current (input) and droplet x-position (output).
      #   Depinning pair has been highlighted, and all IMFs are labeled with their frequency, energy, and classification status.


import pandas as pd
import numpy as np
from PyEMD import EMD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# configs and file paths

#data for slowest episode

#DATA_PATH = '../mbrl_actions_ep20/mbrl_actions_ep201122.csv'
#OUTPUT_PATH = 'Figure3_all_imfs_side_by_side.png'

#data for medium episode

DATA_PATH = '../mbrl_actions_ep20/mbrl_actions_ep200727.csv'
OUTPUT_PATH = 'medium_Figure3_all_imfs_side_by_side.png'

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

# Corner zone
d2c = np.sqrt((x - CORNER[0])**2 + (y - CORNER[1])**2)
stuck_mask = d2c < CORNER_RADIUS
stuck_start = np.where(stuck_mask)[0][0]
stuck_end = np.where(stuck_mask)[0][-1]

# EMD on current
emd1 = EMD()
cur_imfs = emd1.emd(current1 - current1.mean())
n_cur = len(cur_imfs)

# EMD on position
emd2 = EMD()
x_imfs = emd2.emd(x - x.mean())
n_pos = len(x_imfs)

def zc_freq(imf):
    zc = np.sum(np.diff(np.signbit(imf)))
    return (zc / 2) / (N * DT)

# classification from Tests 1 & 2

cur_labels = {
    0: ('NOISE — eliminated by Wu-Huang', '#ff4444'),
    1: ('NOISE — eliminated by Wu-Huang', '#ff4444'),
    2: ('CANDIDATE — passed Wu-Huang + stuck energy', '#00d4ff'),
    3: ('★ DEPINNING — winner of all 5 tests', '#00ff88'),
    4: ('FREE-ACTIVE — eliminated by stuck energy test', '#cc66ff'),
    5: ('FREE-ACTIVE — eliminated by stuck energy test', '#ffd700'),
    6: ('NOISE — eliminated by Wu-Huang', '#ff66cc'),
}

# Position IMFs:
#   0, 1: NOISE
#   2: DEPINNING (0.564 Hz, matches current IMF 3)
pos_labels = {
    0: ('NOISE', '#ff6b6b'),
    1: ('NOISE', '#ffaa44'),
    2: ('★ DEPINNING — 0.564 Hz rocking', '#00ff88'),
    3: ('Slow drift (0.229 Hz)', '#44bbff'),
    4: ('Slow drift (0.123 Hz)', '#dd77ff'),
    5: ('Trend / residual', '#ffdd44'),
}

n_rows = max(n_cur, n_pos) + 1  # +1 for raw signal row
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


# Current
ax = axes[0, 0]
style_ax(ax)
add_corner_shading(ax)
ax.plot(t, current1, color='white', linewidth=0.6)
ax.set_ylabel('mA', color='white', fontsize=16)
ax.set_title('CURRENT (Motor 1)', color='#00d4ff', fontsize=16, fontweight='bold')
ax.text(0.02, 0.82, 'Raw Signal', transform=ax.transAxes,
        color='white', fontsize=16, fontweight='bold')

# Position
ax = axes[0, 1]
style_ax(ax)
add_corner_shading(ax)
ax.plot(t, x, color='white', linewidth=0.6)
ax.set_ylabel('mm', color='white', fontsize=16)
ax.set_title('POSITION (x_mm)', color='#ff6b6b', fontsize=16, fontweight='bold')
ax.text(0.02, 0.82, 'Raw Signal', transform=ax.transAxes,
        color='white', fontsize=16, fontweight='bold')

# Current IMFs 
for i in range(n_cur):
    row = i + 1
    ax = axes[row, 0]
    
    freq = zc_freq(cur_imfs[i])
    energy = np.var(cur_imfs[i])
    label_text, color = cur_labels.get(i, ('', '#888'))
    
    is_winner = (i == 3)
    is_candidate = (i in [2, 3])
    border = '#00ff88' if is_winner else ('#00d4ff' if is_candidate else '#333')
    bw = 2.5 if is_winner else (1.5 if is_candidate else 1)
    
    style_ax(ax, border_color=border, border_width=bw)
    add_corner_shading(ax)
    ax.axhline(0, color='#444', linewidth=0.3)
    
    lw = 1.2 if is_winner else 0.8
    ax.plot(t, cur_imfs[i], color=color, linewidth=lw)
    
    ax.text(0.01, 0.82, f'IMF {i}  |  {freq:.3f} Hz  |  E={energy:.0f}',
            transform=ax.transAxes, color=color, fontsize=12, fontweight='bold')
    
    # Status label on right
    status_color = '#00ff88' if is_winner else (color if is_candidate else '#ff4444')
    ax.text(0.99, 0.82, label_text, transform=ax.transAxes,
            color=status_color, fontsize=10, ha='right',
            fontweight='bold' if is_winner else 'normal')
    
    ax.set_ylabel('mA', color='white', fontsize=16)

# pos IMFs
for i in range(n_pos):
    row = i + 1
    ax = axes[row, 1]
    
    freq = zc_freq(x_imfs[i])
    energy = np.var(x_imfs[i])
    label_text, color = pos_labels.get(i, ('', '#888'))
    
    is_depinning = (i == 2)
    border = '#00ff88' if is_depinning else '#333'
    bw = 2.5 if is_depinning else 1
    
    style_ax(ax, border_color=border, border_width=bw)
    add_corner_shading(ax)
    ax.axhline(0, color='#444', linewidth=0.3)
    
    lw = 1.2 if is_depinning else 0.8
    ax.plot(t, x_imfs[i], color=color, linewidth=lw)
    
    ax.text(0.01, 0.82, f'IMF {i}  |  {freq:.3f} Hz  |  E={energy:.1f}',
            transform=ax.transAxes, color=color, fontsize=12, fontweight='bold')
    
    status_color = '#00ff88' if is_depinning else '#aaa'
    ax.text(0.99, 0.82, label_text, transform=ax.transAxes,
            color=status_color, fontsize=10, ha='right',
            fontweight='bold' if is_depinning else 'normal')
    
    ax.set_ylabel('mm', color='white', fontsize=16)

# Hide extra rows
for row in range(n_cur + 1, n_rows):
    axes[row, 0].set_visible(False)
for row in range(n_pos + 1, n_rows):
    axes[row, 1].set_visible(False)

# Bottom x-labels
axes[n_cur, 0].set_xlabel('Time (s)', color='white', fontsize=16)
axes[n_pos, 1].set_xlabel('Time (s)', color='white', fontsize=16)

# Footer
fig.text(0.5, -0.01,
         'Current IMF 3 (0.653 Hz) ←→ Position IMF 2 (0.564 Hz)\n'
         'Cross-correlation: 0.624  |  Lag: 0.2s  |  '
         'Yellow shading = corner zone',
         ha='center', color='#00ff88', fontsize=11, fontweight='bold',
         linespacing=1.5)

plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"\nSaved: {OUTPUT_PATH}")
