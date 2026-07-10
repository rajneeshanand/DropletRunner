#!/usr/bin/env python3

# Complete Emergent Oscillation Analysis


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
from scipy.signal import butter, filtfilt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.signal import hilbert
from PyEMD import EMD



DATA_DIR = 'mbrl_actions_ep20'          # CSV files
CONTROL_HZ = 20.0                       # Control frequency
DT = 1.0 / CONTROL_HZ                   # 0.05 seconds per step
GOAL_PROGRESS = 150.0                   # mm, threshold for success
MAX_STEPS_TIMEOUT = 590                 # Steps below this = not timeout

# L-shape path coordinates (from waypoints)
L_PATH_X = [87, 87, -26]                # Start -> Corner -> Goal
L_PATH_Y = [-64, 3, 3]
START_POS = (87, -64)
GOAL_POS = (-26, 3)


# STEP 1: Load all episodes

print("=" * 60)
print("Loading episodes from", DATA_DIR)
print("=" * 60)

if not os.path.exists(DATA_DIR):
    print(f"\nERROR: Folder '{DATA_DIR}' not found!")
    print(f"Run: unzip mbrl_actions_ep20.zip -d {DATA_DIR}")
    sys.exit(1)

episodes = []
for f in sorted(os.listdir(DATA_DIR)):
    if not f.endswith('.csv'):
        continue
    df = pd.read_csv(os.path.join(DATA_DIR, f))
    n_steps = len(df)
    final_progress = df['progress'].iloc[-1] if len(df) > 0 else 0
    success = final_progress > GOAL_PROGRESS and n_steps < MAX_STEPS_TIMEOUT
    episodes.append({
        'file': f,
        'n_steps': n_steps,
        'progress': round(final_progress, 1),
        'success': success,
        'df': df,
    })
    status = "SUCCESS" if success else "FAILED"
    print(f"  {f}: {n_steps:4d} steps, progress={final_progress:6.1f}mm  [{status}]")

successful = sorted([e for e in episodes if e['success']], key=lambda e: e['n_steps'])
failed = [e for e in episodes if not e['success']]

print(f"\nTotal: {len(episodes)}, Success: {len(successful)}, Failed: {len(failed)}")


# STEP 2: Select specific episodes


# Fastest success
fast_ep = successful[0]
# Medium success (150-400 steps)
medium_eps = [e for e in successful if 150 < e['n_steps'] < 400]
medium_ep = medium_eps[0] if medium_eps else successful[len(successful) // 2]

# With this to pick a specific file:
#medium_ep = [e for e in episodes if e['file'] == 'mbrl_actions_ep200929.csv'][0]

# Slowest success
slow_ep = successful[-1]
# Failed episode
failed_ep = failed[0] if failed else None

print(f"\nSelected episodes:")
print(f"  Fast:    {fast_ep['file']} ({fast_ep['n_steps']} steps, {fast_ep['n_steps']*DT:.1f}s)")
print(f"  Medium:  {medium_ep['file']} ({medium_ep['n_steps']} steps, {medium_ep['n_steps']*DT:.1f}s)")
print(f"  Slow:    {slow_ep['file']} ({slow_ep['n_steps']} steps, {slow_ep['n_steps']*DT:.1f}s)")
if failed_ep:
    print(f"  Failed:  {failed_ep['file']} ({failed_ep['n_steps']} steps, {failed_ep['n_steps']*DT:.1f}s)")



# STEP 3: Compute statistics


print(f"\n{'='*60}")
print("Computing statistics...")
print(f"{'='*60}")

# Sign change rates for all episodes
for ep in episodes:
    df_ep = ep['df']
    m1 = df_ep['current1'].values
    m2 = df_ep['current2'].values
    if len(m1) > 1:
        ep['sc_m1'] = np.sum(np.diff(np.sign(m1)) != 0) / (len(m1) - 1) * 100
        ep['sc_m2'] = np.sum(np.diff(np.sign(m2)) != 0) / (len(m2) - 1) * 100
    else:
        ep['sc_m1'] = ep['sc_m2'] = 0

succ_rates_m1 = [e['sc_m1'] for e in successful]
succ_rates_m2 = [e['sc_m2'] for e in successful]
print(f"Sign change rate (successful, n={len(successful)}):")
print(f"  Motor 1: {np.mean(succ_rates_m1):.1f} ± {np.std(succ_rates_m1):.1f}%")
print(f"  Motor 2: {np.mean(succ_rates_m2):.1f} ± {np.std(succ_rates_m2):.1f}%")

if failed_ep:
    print(f"Sign change rate (failed, n={len(failed)}):")
    print(f"  Motor 1: {failed_ep['sc_m1']:.1f}%")
    print(f"  Motor 2: {failed_ep['sc_m2']:.1f}%")


# FFT on slow episode
df_med = slow_ep['df']
m1_med = df_med['current1'].values
m2_med = df_med['current2'].values

m1c = m1_med - np.mean(m1_med)
m2c = m2_med - np.mean(m2_med)
n_med = len(m1c)
fft1 = np.fft.rfft(m1c)
fft2 = np.fft.rfft(m2c)
freqs = np.fft.rfftfreq(n_med, d=DT)
power_m1 = np.abs(fft1) ** 2 / n_med
power_m2 = np.abs(fft2) ** 2 / n_med

# Find dominant frequency (exclude < 0.2 Hz to avoid DC-like artifacts)
mask = freqs > 0.2
dom1_idx = np.argmax(power_m1[mask])
dom2_idx = np.argmax(power_m2[mask])
freq_m1 = freqs[mask][dom1_idx]
freq_m2 = freqs[mask][dom2_idx]
pow_m1_dom = power_m1[mask][dom1_idx]
pow_m2_dom = power_m2[mask][dom2_idx]

print(f"\nFFT dominant frequencies (medium episode):")
print(f"  Motor 1: {freq_m1:.2f} Hz (period: {1/freq_m1:.1f}s)")
print(f"  Motor 2: {freq_m2:.2f} Hz (period: {1/freq_m2:.1f}s)")

# FFT across all successful episodes
all_dom_m1 = []
all_dom_m2 = []
for ep in successful:
    df_e = ep['df']
    if len(df_e) > 30:
        m1t = df_e['current1'].values - np.mean(df_e['current1'].values)
        m2t = df_e['current2'].values - np.mean(df_e['current2'].values)
        ft1 = np.fft.rfft(m1t)
        ft2 = np.fft.rfft(m2t)
        fr = np.fft.rfftfreq(len(m1t), d=DT)
        pw1 = np.abs(ft1) ** 2 / len(m1t)
        pw2 = np.abs(ft2) ** 2 / len(m2t)
        msk = fr > 0.2
        if np.any(msk):
            all_dom_m1.append(fr[msk][np.argmax(pw1[msk])])
            all_dom_m2.append(fr[msk][np.argmax(pw2[msk])])

print(f"\nFFT across all successful episodes:")
print(f"  Motor 1: {np.mean(all_dom_m1):.2f} ± {np.std(all_dom_m1):.2f} Hz")
print(f"  Motor 2: {np.mean(all_dom_m2):.2f} ± {np.std(all_dom_m2):.2f} Hz")




print(f"\n{'='*60}")
print("Generating figure...")
print(f"{'='*60}")

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 24,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})


fig = plt.figure(figsize=(14, 18))


# Plot (a): All successful trajectories overlaid on L-path


ax = fig.add_subplot(4, 2, 1)
ax.text(-0.17, 1.15, 'a', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top')

# Draw L-shape path
ax.plot([87, 87], [-64, 3], '-', color='black', linewidth=5, alpha=0.3, zorder=1)
ax.plot([87, -26], [3, 3], '-', color='black', linewidth=5, alpha=0.3, zorder=1, label='L-path')

# Plot all successful trajectories (light)
cmap = plt.cm.Blues
for i, ep in enumerate(successful):
    df = ep['df']
    color = cmap(0.3 + 0.5 * i / len(successful))
    ax.plot(df['x_mm'], df['y_mm'], '-', color=color, linewidth=1.0, alpha=0.4, zorder=2)

# Highlight fast, medium, slow
ax.plot(fast_ep['df']['x_mm'], fast_ep['df']['y_mm'], '-', color='#2166AC',
        linewidth=3, zorder=4, label=f'Fast ({fast_ep["n_steps"]} steps)')
ax.plot(medium_ep['df']['x_mm'], medium_ep['df']['y_mm'], '-', color='#D6604D',
        linewidth=3, zorder=4, label=f'Medium ({medium_ep["n_steps"]} steps)')
ax.plot(slow_ep['df']['x_mm'], slow_ep['df']['y_mm'], '-', color='#1B7837',
        linewidth=2, zorder=3, label=f'Slow ({slow_ep["n_steps"]} steps)')

# Start and Goal markers
ax.plot(START_POS[0], START_POS[1], 'o', color='green', markersize=10, zorder=10)
ax.plot(GOAL_POS[0], GOAL_POS[1], '*', color='red', markersize=14, zorder=10)
ax.text(START_POS[0] + 5, START_POS[1], 'Start', fontsize=10, fontweight='bold', color='green')
ax.text(GOAL_POS[0], GOAL_POS[1] - 6, 'Goal', ha='center', fontsize=10, fontweight='bold', color='red')

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')

ax.set_title(f'MBRL trajectories: all {len(successful)} successful episodes')

ax.legend(loc='lower left', frameon=False, fontsize=18)
#ax.grid(alpha=0.15)
ax.set_aspect('equal')


# Panel (b): Failed trajectory colored by time


ax = fig.add_subplot(4, 2, 2)
ax.text(-0.2, 1.25, 'b', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top')

# Draw L-shape path
ax.plot([87, 87], [-64, 3], '-', color='black', linewidth=5, alpha=0.3, zorder=1)
ax.plot([87, -26], [3, 3], '-', color='black', linewidth=5, alpha=0.3, zorder=1)

if failed_ep:
    df_f = failed_ep['df']
    n_f = len(df_f)

    # Color each segment by time
    for i in range(n_f - 1):
        color = plt.cm.hot(i / n_f)
        ax.plot(df_f['x_mm'].iloc[i:i + 2], df_f['y_mm'].iloc[i:i + 2],
                '-', color=color, linewidth=3.0)

    # Start marker
    ax.plot(df_f['x_mm'].iloc[0], df_f['y_mm'].iloc[0], 'o', color='green',
            markersize=10, zorder=10)
    # Goal (faded — not reached)
    ax.plot(GOAL_POS[0], GOAL_POS[1], '*', color='red', markersize=14, zorder=10, alpha=0.3)

    # Annotate stuck point
    stuck_x = df_f['x_mm'].iloc[-50:].mean()
    stuck_y = df_f['y_mm'].iloc[-50:].mean()
    ax.annotate('Timeout here', xy=(stuck_x, stuck_y),
                xytext=(stuck_x - 40, stuck_y - 25),
                fontsize=18, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=2.0))

    # Colorbar for time
    sm = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=plt.Normalize(0, 30))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Time (s)', shrink=1.0)

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')

ax.set_title(f'Failed trajectory')

ax.set_aspect('equal')


# Panel (c): Motor commands - fast successful episode


ax = fig.add_subplot(4, 2, 3)
ax.text(-0.15, 1.15, 'c', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top')

df_fast = fast_ep['df']
t_fast = df_fast['step'].values * DT

ax.plot(t_fast, df_fast['current1'].values, '-', color='#2166AC',
        linewidth=1.0, alpha=0.8, label='Motor 1')
ax.plot(t_fast, df_fast['current2'].values, '-', color='#D6604D',
        linewidth=1.0, alpha=0.8, label='Motor 2')
ax.axhline(y=0, color='gray', linewidth=0.5)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Motor Current (mA)')

ax.set_title(f'Motor commands: fast success ({fast_ep["n_steps"]} steps, {fast_ep["n_steps"]*DT:.1f}s)')

ax.legend(loc='upper right', frameon=False, fontsize=18)
ax.set_ylim(-180, 180)
#ax.grid(alpha=0.15)


# Panel (d): Motor commands - failed episode

ax = fig.add_subplot(4, 2, 4)
ax.text(-0.15, 1.15, 'd', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top')

if failed_ep:
    t_fail = df_f['step'].values * DT

    ax.plot(t_fail, df_f['current1'].values, '-', color='#2166AC',
            linewidth=0.5, alpha=0.7, label='Motor 1')
    ax.plot(t_fail, df_f['current2'].values, '-', color='#D6604D',
            linewidth=0.5, alpha=0.7, label='Motor 2')
    ax.axhline(y=0, color='gray', linewidth=0.5)

    # Find and shade stagnation region
    progress_f = df_f['progress'].values
    for i in range(len(progress_f) - 1):
        if progress_f[i] > 50 and abs(progress_f[i + 1] - progress_f[i]) < 0.5:
            ax.axvspan(t_fail[i], t_fail[-1], alpha=0.06, color='red')
            ax.text(t_fail[-1] * 0.55, 160, 'Oscillatory depinning attempts',
                    fontsize=14, color='black', ha='center', style='italic')
            break

ax.set_xlabel('Time (s)')
ax.set_ylabel('Motor Current (mA)')

ax.set_title(f'Motor commands: stuck at corner (600 steps, 30s)')

ax.legend(loc='upper right', frameon=False, fontsize=18)
ax.set_ylim(-180, 180)


# ---- use the slowest successful episode (same one as before) ----
df_slow = slow_ep['df']
x_s = df_slow['x_mm'].values
y_s = df_slow['y_mm'].values
N_s = len(df_slow)


#df_med = medium_ep['df']
#x_s = df_med['x_mm'].values
#y_s = df_med['y_mm'].values
#N_s = len(df_med)

t_s = np.arange(N_s) * DT
CORNER = (87.0, 3.0)            # corner coordinate


# ---- Hilbert instantaneous frequency & amplitude of depinning mode ----
# IMF selection: Wu-Huang significance test + stuck-phase energy + position frequency cross-validation

def _position_depinning_freq():
    x_sig = x_s - x_s.mean()
    x_imfs = EMD().emd(x_sig)
    x_var = np.var(x_sig)
    
    # Wu-Huang on position
    np.random.seed(99)
    x_mc = [[] for _ in range(len(x_imfs))]
    for trial in range(500):
        wn = np.random.randn(N_s) * np.sqrt(x_var)
        try:
            wn_imfs = EMD().emd(wn)
        except:
            continue
        for idx in range(min(len(wn_imfs), len(x_imfs))):
            x_mc[idx].append(np.var(wn_imfs[idx]))
    
    # Find fastest significant IMF above 0.1 Hz
    for i in range(len(x_imfs) - 1):
        e = np.var(x_imfs[i])
        freq = (np.sum(np.diff(np.signbit(x_imfs[i]))) / 2) / (N_s * DT)
        if len(x_mc[i]) > 30:
            p95 = np.percentile(x_mc[i], 95)
            if e > p95 and freq > 0.1:
                return freq
    return 0.5  # safe fallback


def hilbert_mode(sig, n_mc=1000, seed=None):
    s = sig - sig.mean()
    n = len(s)
    imfs = EMD().emd(s)
    
    # Step 1: Wu-Huang significance test
    sig_energy = np.var(s)
    if seed is not None:
        np.random.seed(seed)
    
    mc_e = [[] for _ in range(len(imfs))]
    for trial in range(n_mc):
        wn = np.random.randn(n) * np.sqrt(sig_energy)
        try:
            wn_imfs = EMD().emd(wn)
        except:
            continue
        for idx in range(min(len(wn_imfs), len(imfs))):
            mc_e[idx].append(np.var(wn_imfs[idx]))
    
    significant = []
    for i in range(len(imfs) - 1):  # skip residual
        e = np.var(imfs[i])
        if len(mc_e[i]) > 30:
            p95 = np.percentile(mc_e[i], 95)
            if e > p95:
                fm = (np.sum(np.diff(np.signbit(imfs[i]))) / 2) / (n * DT)
                significant.append((i, fm))
    
    # Handle based on number of survivors
    if len(significant) == 0:
        # Fallback: no significant IMFs found
        best_idx = min(3, len(imfs) - 1)
        best = imfs[best_idx]
    
    elif len(significant) == 1:
        # Single survivor — use directly (e.g. current2)
        best_idx = significant[0][0]
        best = imfs[best_idx]
    
    else:
        # Multiple survivors: apply stuck energy + frequency match
        
        # Step 2: Stuck-phase energy ratio
        d2c_local = np.sqrt((x_s - CORNER[0])**2 + (y_s - CORNER[1])**2)
        stuck_idx = np.where(d2c_local < 30.0)[0]
        
        stuck_active = []
        for i, fm in significant:
            imf = imfs[i]
            e_stuck = np.mean(imf[stuck_idx[0]:stuck_idx[-1]]**2)
            free_samp = np.concatenate([imf[:stuck_idx[0]], imf[stuck_idx[-1]:]])
            e_free = np.mean(free_samp**2) if len(free_samp) > 0 else 1e-6
            ratio = e_stuck / e_free
            if ratio > 1.0:
                stuck_active.append((i, fm, ratio))
        
        if len(stuck_active) == 0:
            # No stuck-active IMFs, use first significant
            best_idx = significant[0][0]
            best = imfs[best_idx]
        
        elif len(stuck_active) == 1:
            # Single stuck-active survivor
            best_idx = stuck_active[0][0]
            best = imfs[best_idx]
        
        else:
            # Step 3: Frequency match to position
            best_dist = 1e9
            best_idx = stuck_active[0][0]
            for i, fm, ratio in stuck_active:
                dist = abs(fm - _pos_freq)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            best = imfs[best_idx]
    
    # Step 4: Hilbert transform
    z = hilbert(best)
    amp = np.abs(z)
    phase = np.unwrap(np.angle(z))
    f = np.gradient(phase, DT) / (2 * np.pi)
    f[(amp < 0.2 * np.median(amp)) | (f < 0) | (f > 1.5)] = np.nan
    w = np.hanning(9); w /= w.sum()
    f = np.convolve(np.nan_to_num(f, nan=np.nanmedian(f)), w, mode='same')  
    return f, amp


# Compute position depinning frequency once (ground truth)
_pos_freq = _position_depinning_freq()

# Extract depinning frequency and amplitude for both motors
f1, amp1 = hilbert_mode(df_slow['current1'].values.astype(float), seed=42)
f2, amp2 = hilbert_mode(df_slow['current2'].values.astype(float), seed=77)

# ---- phase windows (corner = within 1 cm of the corner) ----
d2c = np.sqrt((x_s - CORNER[0]) ** 2 + (y_s - CORNER[1]) ** 2)
zc = np.where(d2c < 30.0)[0]
i1p, i2p = zc[0], zc[-1]

PHASES = [(0, i1p, '#4C9F70', ''),
          (i1p, i2p, '#E07A3E', ''),
          (i2p, N_s - 1, '#5B7DB1', '')]


          

# Panel (e): emergent vibration FREQUENCY  (top-left)

ax = fig.add_subplot(4, 2, 5)
ax.text(-0.15, 1.15, 'e', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top')
for a, b, col, nm in PHASES:
    ax.axvspan(t_s[a], t_s[b], color=col, alpha=0.12)
    ax.text((t_s[a] + t_s[b]) / 2, 1.42, nm, ha='center', color=col, fontsize=8, fontweight='bold')
ax.plot(t_s, f1, '-', color='#2166AC', linewidth=2.0, label='Motor 1')
ax.plot(t_s, f2, '-', color='#D6604D', linewidth=2.0, label='Motor 2')
ax.set_ylim(0, 1.5); ax.set_xlim(0, t_s[-1])
ax.set_xlabel('Time (s)'); ax.set_ylabel('Vibration frequency (Hz)')
ax.set_title('Emergent vibration: frequency')
ax.legend(loc='upper right', frameon=False, fontsize=18)


# Panel (f): emergent vibration STRENGTH  (pairs with e)

ax = fig.add_subplot(4, 2, 6)
ax.text(-0.15, 1.15, 'f', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top')
for a, b, col, nm in PHASES:
    ax.axvspan(t_s[a], t_s[b], color=col, alpha=0.12)
ax.plot(t_s, amp1, '-', color='#2166AC', linewidth=2.0, label='Motor 1')
ax.plot(t_s, amp2, '-', color='#D6604D', linewidth=2.0, label='Motor 2')
ax.set_xlim(0, t_s[-1])
ax.set_xlabel('Time (s)'); ax.set_ylabel('Vibration strength (mA)')
ax.set_title('Emergent vibration: strength')
ax.legend(loc='upper right', frameon=False, fontsize=18)
 

# Panel (g): DWELL TIME map  (top-right)


ax = fig.add_subplot(4, 2, 7)
ax.text(-0.2, 1.15, 'g', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top')
ax.plot([87, 87], [-64, 3], '-', color='gray', linewidth=5, alpha=0.25)
ax.plot([87, -26], [3, 3], '-', color='gray', linewidth=5, alpha=0.25)
dwell = np.array([np.sum(np.sqrt((x_s - x_s[i]) ** 2 + (y_s - y_s[i]) ** 2) < 5.0) * DT
                  for i in range(N_s)])
dn = Normalize(vmin=0, vmax=dwell.max())
for i in range(N_s - 1):
    ax.plot(x_s[i:i + 2], y_s[i:i + 2], '-', color=plt.cm.hot(dn(dwell[i])), linewidth=3.0)
ax.plot(x_s[0], y_s[0], 'o', color='green', markersize=10, zorder=10)
ax.plot(x_s[-1], y_s[-1], '*', color='red', markersize=10, alpha=0.6, zorder=10)
sm = ScalarMappable(cmap='hot', norm=dn); sm.set_array([])
fig.colorbar(sm, ax=ax, label='seconds', shrink=1.0)
ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); 
#ax.set_aspect('equal')
ax.set_aspect('equal', adjustable='datalim')
ax.set_title('Dwell time')
 

 
# Step 4: ACCELERATION map  (bottom-right)


ax = fig.add_subplot(4, 2, 8)
ax.text(-0.15, 1.15, 'h', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top')
ax.plot([87, 87], [-64, 3], '-', color='gray', linewidth=5, alpha=0.25)
ax.plot([87, -26], [3, 3], '-', color='gray', linewidth=5, alpha=0.25)
dx = np.diff(x_s) / DT; dy = np.diff(y_s) / DT
axx = np.diff(dx) / DT; ayy = np.diff(dy) / DT
acc = np.convolve(np.sqrt(axx ** 2 + ayy ** 2), np.ones(5) / 5, mode='same')
acc_full = np.zeros(N_s); acc_full[1:-1] = acc; acc_full[0] = acc_full[1]; acc_full[-1] = acc_full[-2]
an = Normalize(vmin=0, vmax=np.percentile(acc_full, 95))
for i in range(N_s - 1):
    ax.plot(x_s[i:i + 2], y_s[i:i + 2], '-', color=plt.cm.inferno(an(acc_full[i])), linewidth=3.0)
ax.plot(x_s[0], y_s[0], 'o', color='green', markersize=10, zorder=10)
ax.plot(x_s[-1], y_s[-1], '*', color='red', markersize=10, alpha=0.6, zorder=10)
sm2 = ScalarMappable(cmap='inferno', norm=an); sm2.set_array([])
fig.colorbar(sm2, ax=ax, label='mm/s$^2$', shrink=1.0)
ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)');
#ax.set_aspect('equal')
ax.set_aspect('equal', adjustable='datalim')
ax.set_title('Acceleration')
 


#plt.tight_layout()
plt.tight_layout(pad=2.0)
plt.savefig('fig_complete_emergent_analysis.png', bbox_inches='tight', dpi=600)

# Save as PDF for high-quality printing
plt.savefig('fig_complete_emergent_analysis.pdf', bbox_inches='tight', dpi=600)




