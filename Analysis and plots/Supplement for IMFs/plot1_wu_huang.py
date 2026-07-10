
#Plot 1: Wu-Huang Statistical Significance Test

#PURPOSE: Identify which IMFs contain real signal vs noise.
#METHOD:  Compare each IMF's energy against 1000 white noise realizations at the same timescale.
#RESULT:  IMFs 0, 1, 6 eliminated as noise. IMFs 2, 3, 4, 5 survive as real signal.
#Reference: Wu & Huang (2004), Proc. R. Soc. Lond. A, 460, 1597-1611

import pandas as pd
import numpy as np
from PyEMD import EMD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Configs and file paths
_HERE = Path(__file__).parent

#data for slowest episode

#DATA_PATH = _HERE / '../mbrl_actions_ep20/mbrl_actions_ep201122.csv'
#OUTPUT_PATH = _HERE / 'Figure1_wu_huang.png'

#data for medium episode

DATA_PATH = _HERE / '../mbrl_actions_ep20/mbrl_actions_ep200727.csv'
OUTPUT_PATH = _HERE / 'medium_Figure1_wu_huang.png'

DT = 0.05
N_MC = 1000
BG = '#0e1117'


df = pd.read_csv(DATA_PATH)
N = len(df)
current1 = df['current1'].values.astype(float)

emd = EMD()
cur_sig = current1 - current1.mean()
imfs = emd.emd(cur_sig)
n_imfs = len(imfs)

def zc_freq(imf):
    zc = np.sum(np.diff(np.signbit(imf)))
    return (zc / 2) / (N * DT)

freqs = [zc_freq(imf) for imf in imfs]
energies = [np.var(imf) for imf in imfs]

#Monte Carlo: EMD on white noise 
sig_energy = np.var(cur_sig)
np.random.seed(42)
mc_e = [[] for _ in range(12)]

for trial in range(N_MC):
    wn = np.random.randn(N) * np.sqrt(sig_energy)
    emd_mc = EMD()
    try:
        wn_imfs = emd_mc.emd(wn)
    except:
        continue
    for idx, imf_wn in enumerate(wn_imfs):
        mc_e[idx].append(np.var(imf_wn))

#classification of each IMFs
verdicts = []
noise_99 = []
noise_means = []
for i in range(n_imfs):
    if len(mc_e[i]) > 30:
        p99 = np.percentile(mc_e[i], 99)
        nm = np.mean(mc_e[i])
        noise_99.append(p99)
        noise_means.append(nm)
        if energies[i] > p99:
            verdicts.append('SIGNAL')
        else:
            verdicts.append('NOISE')
    else:
        noise_99.append(0)
        noise_means.append(0)
        verdicts.append('?')

# results
print(f"{'IMF':<5} {'Freq (Hz)':<11} {'Energy (E)':<13} {'Noise (N) 99%':<13} {'Verdict'}")
print("-" * 55)
for i in range(n_imfs):
    print(f"  {i:<3} {freqs[i]:<11.3f} {energies[i]:<13.1f} {noise_99[i]:<13.1f} {verdicts[i]}")

# plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), facecolor=BG)

#Left: Bar chart comparison
ax1.set_facecolor(BG)
x_pos = np.arange(n_imfs)
width = 0.35

signal_colors = ['#00ff88' if v == 'SIGNAL' else '#ff4444' for v in verdicts]

bars_you = ax1.bar(x_pos - width/2, energies, width,
                    color=signal_colors, edgecolor='#333', alpha=0.85,
                    label='IMF Energy (E)')
bars_noise = ax1.bar(x_pos + width/2, noise_99, width,
                      color='#444', edgecolor='#666', alpha=0.7,
                      label='Noise (N) 99th percentile')

ax1.set_xlabel('IMF Index', color='white', fontsize=11)
ax1.set_ylabel('Energy (mA²)', color='white', fontsize=11)
ax1.set_title('Is IMF energy above the noise threshold?',
              color='white', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'IMF {i}\n{freqs[i]:.2f} Hz' for i in range(n_imfs)], fontsize=8)
ax1.tick_params(colors='white')
ax1.legend(fontsize=9, facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')

# reason labels
reasons = {
    0: 'NOISE\nE 2966 < N 8440\nBang-bang switching',
    1: 'NOISE\nE 1762 < N 3348',
    2: 'SIGNAL ✓\nE 3406 > N 1848',
    3: 'SIGNAL ✓\nE 1272 > N 1079',
    4: 'SIGNAL ✓\nE 750 > N 633',
    5: 'SIGNAL ✓\nE 517 > N 424',
    6: 'NOISE\nE 67 < N 293\nSlow DC drift',
}

for i, rect in enumerate(bars_you):
    color = '#00ff88' if verdicts[i] == 'SIGNAL' else '#ff4444'
    y_pos = max(energies[i], noise_99[i]) + 200
    ax1.text(i, y_pos, reasons[i], ha='center', va='bottom',
             fontsize=6.5, color=color, fontweight='bold', linespacing=1.3)

for spine in ax1.spines.values():
    spine.set_edgecolor('#333')

# Right: Wu-Huang ln(E) vs ln(T) plot 
# ax2.set_facecolor(BG)

# Compute ln(E) and ln(T) for noise reference
mc_lnE = []
mc_lnT = []
mc_upper = []
mc_lower = []
for i in range(n_imfs):
    if len(mc_e[i]) > 30:
        # Period from counting maxima
        n_max = sum(1 for k in range(1, len(imfs[i])-1)
                    if imfs[i][k] > imfs[i][k-1] and imfs[i][k] > imfs[i][k+1])
        period = N / n_max if n_max > 0 else N
        mc_lnT.append(np.log(period))
        mc_lnE.append(np.mean(np.log(np.array(mc_e[i]))))
        mc_upper.append(np.percentile(np.log(np.array(mc_e[i])), 99))
        mc_lower.append(np.percentile(np.log(np.array(mc_e[i])), 1))

mc_lnT = np.array(mc_lnT)
mc_lnE = np.array(mc_lnE)
mc_upper = np.array(mc_upper)
mc_lower = np.array(mc_lower)

si = np.argsort(mc_lnT)
ax2.plot(mc_lnT[si], mc_lnE[si], 'w-', linewidth=2, label='White noise mean')
ax2.fill_between(mc_lnT[si], mc_lower[si], mc_upper[si],
                  color='#00d4ff', alpha=0.12, label='99% confidence band')
ax2.plot(mc_lnT[si], mc_upper[si], '--', color='#00d4ff', linewidth=1, alpha=0.6)
ax2.plot(mc_lnT[si], mc_lower[si], '--', color='#00d4ff', linewidth=1, alpha=0.6)

# Plot each IMF
for i in range(n_imfs):
    n_max = sum(1 for k in range(1, len(imfs[i])-1)
                if imfs[i][k] > imfs[i][k-1] and imfs[i][k] > imfs[i][k+1])
    period = N / n_max if n_max > 0 else N
    lt = np.log(period)
    le = np.log(energies[i])
    
    color = '#00ff88' if verdicts[i] == 'SIGNAL' else '#ff4444'
    marker = '★' if verdicts[i] == 'SIGNAL' else '●'                   # please use window + . to get such symbols of stars and dots
    size = 150 if verdicts[i] == 'SIGNAL' else 80
    m = '*' if verdicts[i] == 'SIGNAL' else 'o'
    
    ax2.scatter(lt, le, c=color, marker=m, s=size, zorder=5,
                edgecolors='white', linewidth=0.5)
    ax2.annotate(f'IMF {i}\n({freqs[i]:.2f} Hz)',
                 (lt, le), textcoords='offset points', xytext=(10, 5),
                 fontsize=7.5, color='white', alpha=0.85)

ax2.set_xlabel('ln(Mean Period)  [samples]', color='white', fontsize=11)
ax2.set_ylabel('ln(Energy)', color='white', fontsize=11)
ax2.set_title('Wu-Huang Plot\nAbove the band = real signal, inside = noise',
              color='white', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8, facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('#333')

fig.suptitle('Test 1: Wu-Huang Statistical Significance — IMFs 0, 1, 6 eliminated as noise\n'
             'Survivors: IMFs 2, 3, 4, 5 (energy above 99% noise threshold)',
             color='white', fontsize=14, fontweight='bold', y=1.04)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"\nSaved: {OUTPUT_PATH}")
