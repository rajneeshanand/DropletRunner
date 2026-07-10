
#Plot 2: Stuck-Phase Energy Concentration

#PURPOSE: Among Wu-Huang survivors (IMFs 2,3,4,5), which are active during depinning (corner zone) vs free motion?
#METHOD:  Split each IMF into stuck portion and free portion and compute energy per sample in each. Compare via ratio.
        # Ratio > 1 = stuck-active = depinning candidate. Ratio < 1 = free-active = not depinning.
# RESULT:  IMFs 4, 5 eliminated (active during free motion). IMFs 2, 3 survive as depinning candidates.


import pandas as pd
import numpy as np
from PyEMD import EMD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# configs and file paths

#data for slowest episode

#DATA_PATH = '../mbrl_actions_ep20/mbrl_actions_ep201122.csv'
#OUTPUT_PATH = 'Figure2_stuck_energy.png'

#data for medium episode

DATA_PATH = '../mbrl_actions_ep20/mbrl_actions_ep200727.csv'
OUTPUT_PATH = 'medium_Figure2_stuck_energy.png'


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

# stuck zone
d2c = np.sqrt((x - CORNER[0])**2 + (y - CORNER[1])**2)
stuck_mask = d2c < CORNER_RADIUS
stuck_start = np.where(stuck_mask)[0][0]
stuck_end = np.where(stuck_mask)[0][-1]

# EMD
emd = EMD()
imfs = emd.emd(current1 - current1.mean())

def zc_freq(imf):
    zc = np.sum(np.diff(np.signbit(imf)))
    return (zc / 2) / (N * DT)

# only Wu-Huang survivors 
target = [2, 3, 4, 5]

data = {}
for i in target:
    imf = imfs[i]
    stuck_portion = imf[stuck_start:stuck_end]
    free_portion = np.concatenate([imf[:stuck_start], imf[stuck_end:]])
    
    e_stuck = np.mean(stuck_portion ** 2)
    e_free = np.mean(free_portion ** 2)
    ratio = e_stuck / e_free
    
    data[i] = {
        'freq': zc_freq(imf),
        'e_stuck': e_stuck,
        'e_free': e_free,
        'ratio': ratio,
        'candidate': ratio > 1.0,
        'n_stuck': len(stuck_portion),
        'n_free': len(free_portion),
    }

# results
print(f"Stuck phase: steps {stuck_start}-{stuck_end} "
      f"({stuck_start*DT:.1f}s - {stuck_end*DT:.1f}s)")
print(f"  Stuck samples: {stuck_end - stuck_start}")
print(f"  Free samples:  {stuck_start + (N - stuck_end)}\n")

for i in target:
    d = data[i]
    status = "CANDIDATE" if d['candidate'] else "ELIMINATED"
    print(f"IMF {i} ({d['freq']:.3f} Hz):")
    print(f"  E_stuck = {d['e_stuck']:.1f} mA²  ({d['n_stuck']} samples)")
    print(f"  E_free  = {d['e_free']:.1f} mA²  ({d['n_free']} samples)")
    print(f"  Ratio   = {d['ratio']:.2f}  → {status}\n")

# plotting
fig = plt.figure(figsize=(16, 16), facecolor=BG)
gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 2], hspace=0.4)

colors_imf = {2: '#00d4ff', 3: '#00ff88', 4: '#cc66ff', 5: '#ffd700'}

# reasons for elimination/survival
reasons = {
    2: ('CANDIDATE',
        f'E_stuck (3593) > E_free (1695)\n'
        f'Ratio = 2.12 → 2.1x more active in corner\n'
        f'This IMF concentrates energy during depinning'),
    3: ('CANDIDATE',
        f'E_stuck (1322) > E_free (817)\n'
        f'Ratio = 1.62 → 1.6x more active in corner\n'
        f'This IMF concentrates energy during depinning'),
    4: ('ELIMINATED',
        f'E_stuck (678) < E_free (1447)\n'
        f'Ratio = 0.47 → 2.1x more active in FREE phase\n'
        f'Active during navigation, not depinning'),
    5: ('ELIMINATED',
        f'E_stuck (337) < E_free (2199)\n'
        f'Ratio = 0.15 → 6.5x more active in FREE phase\n'
        f'Slow approach/exit drift, not depinning'),
}

# Rows 0-3: IMF waveforms
for j, i in enumerate(target):
    ax = fig.add_subplot(gs[j])
    ax.set_facecolor(BG)
    
    imf = imfs[i]
    d = data[i]
    c = colors_imf[i]
    is_cand = d['candidate']
    
    # Shade regions
    ax.axvspan(0, stuck_start*DT, color='#ff4444', alpha=0.06)
    ax.axvspan(stuck_start*DT, stuck_end*DT, color='#00ff88', alpha=0.06)
    ax.axvspan(stuck_end*DT, N*DT, color='#ff4444', alpha=0.06)
    ax.axvline(stuck_start*DT, color='#ffd700', linewidth=1.2, 
               linestyle='--', alpha=0.5)
    ax.axvline(stuck_end*DT, color='#ffd700', linewidth=1.2, 
               linestyle='--', alpha=0.5)
    ax.axhline(0, color='#444', linewidth=0.3)
    
    ax.plot(t, imf, color=c, linewidth=0.9)
    
    # IMF label
    ax.text(0.01, 0.82, f'IMF {i}  |  {d["freq"]:.3f} Hz',
            transform=ax.transAxes, color=c, fontsize=12, fontweight='bold')
    
    # Status + reason
    status_text, reason_text = reasons[i]
    status_color = '#00ff88' if is_cand else '#ff4444'
    ax.text(0.99, 0.82, status_text, transform=ax.transAxes,
            color=status_color, fontsize=10, fontweight='bold', ha='right')
    ax.text(0.99, 0.15, reason_text, transform=ax.transAxes,
            color=status_color, fontsize=8.5, ha='right', va='bottom',
            linespacing=1.4, fontstyle='italic')
    
    # Energy annotations in each region

    # Left free region
    if stuck_start > 5:
        ax.text(stuck_start*DT*0.4, 0.5,
                f'E = {d["e_free"]:.0f} mA²/sample',
                transform=ax.get_xaxis_transform(),
                color='white', fontsize=10, ha='center', va='center')
    
    # Stuck region
    mid_stuck_time = (stuck_start + stuck_end) / 2 * DT
    ax.text(mid_stuck_time, 0.5,
            f'E = {d["e_stuck"]:.0f} mA²/sample',
            transform=ax.get_xaxis_transform(),
            color='white', fontsize=10, ha='center', va='center')
    
    # Right free region
    if (N - stuck_end) > 5:
        right_mid = (stuck_end + N) / 2 * DT
        ax.text(right_mid, 0.5,
                f'E = {d["e_free"]:.0f} mA²/sample',
                transform=ax.get_xaxis_transform(),
                color='white', fontsize=10, ha='center', va='center')
    
    ax.set_ylabel('mA', color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=8)
    
    border_color = '#00ff88' if is_cand else '#ff4444'
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(2)
    
    if j == 0:
        ax.text(stuck_start*DT*0.4 / t[-1], 1.08, 'Free',
                transform=ax.transAxes, color='#ff6b6b', fontsize=12, ha='center')
        ax.text(0.5, 1.08, 'corner zone',
                transform=ax.transAxes, color='#66ff66', fontsize=12, ha='center')
        ax.text(1 - (N - stuck_end)*DT*0.4 / t[-1], 1.08, 'Free',
                transform=ax.transAxes, color='#ff6b6b', fontsize=12, ha='center')
    
    if j == 3:
        ax.set_xlabel('Time (s)', color='white', fontsize=14)

# Row 4: Bar chart
ax_bar = fig.add_subplot(gs[4])
ax_bar.set_facecolor(BG)

x_pos = np.arange(4)
width = 0.35

stuck_vals = [data[i]['e_stuck'] for i in target]
free_vals = [data[i]['e_free'] for i in target]

bars_s = ax_bar.bar(x_pos - width/2, stuck_vals, width,
                     color='#00ff88', edgecolor='#333', alpha=0.8,
                     label='Energy/sample INSIDE corner (stuck)')
bars_f = ax_bar.bar(x_pos + width/2, free_vals, width,
                     color='#ff4444', edgecolor='#333', alpha=0.8,
                     label='Energy/sample OUTSIDE corner (free)')

# Ratio annotations
for j, i in enumerate(target):
    d = data[i]
    y_top = max(d['e_stuck'], d['e_free']) + 150
    
    if d['candidate']:
        txt = (f'Ratio = {d["ratio"]:.2f}x\n'
               f'Stuck > Free\n CANDIDATE')
        color = '#00ff88'
    else:
        txt = (f'Ratio = {d["ratio"]:.2f}x\n'
               f'Free > Stuck\n ELIMINATED')
        color = '#ff4444'
    
    ax_bar.text(j, y_top, txt, ha='center', va='bottom',
                fontsize=10, color=color, fontweight='bold', linespacing=1.3)

# Dotted connector showing which bar is bigger
for j, i in enumerate(target):
    d = data[i]
    s_val = d['e_stuck']
    f_val = d['e_free']
    bigger = max(s_val, f_val)
    smaller = min(s_val, f_val)
    
    # Arrow from smaller to bigger
    if s_val > f_val:
        ax_bar.annotate('', xy=(j - width/2, bigger),
                        xytext=(j + width/2, f_val),
                        arrowprops=dict(arrowstyle='->', color='#00ff88',
                                       linewidth=1.5, linestyle='--'))
    else:
        ax_bar.annotate('', xy=(j + width/2, bigger),
                        xytext=(j - width/2, s_val),
                        arrowprops=dict(arrowstyle='->', color='#ff4444',
                                       linewidth=1.5, linestyle='--'))

ax_bar.set_ylabel('Energy per sample (mA²)', color='white', fontsize=11)
ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels([f'IMF {i}\n{data[i]["freq"]:.3f} Hz' for i in target],
                        fontsize=10, color='white')
ax_bar.tick_params(colors='white', labelsize=9)
ax_bar.legend(fontsize=10, facecolor='#1a1a2e', edgecolor='#444',
              labelcolor='white', loc='upper right')
for spine in ax_bar.spines.values():
    spine.set_edgecolor('#333')

fig.suptitle('Test 2: Stuck-Phase Energy Concentration\n'
             'Among Wu-Huang survivors (IMFs 2, 3, 4, 5) — which are active during depinning?\n'
             'Energy per sample = mean of squared IMF values  |  '
             'Ratio = stuck energy ÷ free energy  |  Ratio > 1 = depinning candidate',
             color='white', fontsize=13, fontweight='bold', y=1.02)

plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"\nSaved: {OUTPUT_PATH}")
