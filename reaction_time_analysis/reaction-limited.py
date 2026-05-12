import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.style.use('plotstylefile.mplstyle')
d = 10 #code distance 
reaction_depths = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 4, 4, 4, 3, 4, 3, 3, 4, 3, 4, 4, 4, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 2, 2, 2, 3, 2, 3, 3, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2, 2, 2, 2, 3, 2, 3, 3, 2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 3, 2, 2, 3, 2, 2, 3, 3, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
# tau_r sweep from 0.01us to 10ms
tau_r = np.logspace(1.5, 3, 100) # us
print(tau_r)
# exit()
# tau_c sweep from 0.01us to 10ms
tau_c = np.logspace(-2, 2, 100) # us


stalling_time_grid = np.zeros((len(tau_r), len(tau_c)))
for i in range(len(tau_r)):
    for j in range(len(tau_c)):
        stalling_time = sum((((reaction_depths[k]*tau_r[i]) + 2*tau_c[j]) - d*tau_c[j]) for k in range(len(reaction_depths)))
        if stalling_time < 0:
            stalling_time = 0
        stalling_time_grid[i, j] = stalling_time

# Heatmap: tau_c (x) vs tau_r (y) for stalling time
fig, ax = plt.subplots(figsize=(8, 6))
# Edges for pcolormesh (one extra point per dimension)
tau_r_edges = np.concatenate((tau_r[:1] / np.sqrt(tau_r[1] / tau_r[0]),
                              np.sqrt(tau_r[:-1] * tau_r[1:]),
                              [tau_r[-1] * np.sqrt(tau_r[-1] / tau_r[-2])]))
tau_c_edges = np.concatenate((tau_c[:1] / np.sqrt(tau_c[1] / tau_c[0]),
                              np.sqrt(tau_c[:-1] * tau_c[1:]),
                              [tau_c[-1] * np.sqrt(tau_c[-1] / tau_c[-2])]))
pc = ax.pcolormesh(tau_c_edges, tau_r_edges, stalling_time_grid, shading='flat', cmap='viridis')
ax.set_xscale('log')
ax.set_yscale('log')
# Boundary where stalling = 0: sum(rd)*tau_r = (d-2)*tau_c*N  =>  tau_r = (d-2)*tau_c*N/sum(rd)
sum_rd = sum(reaction_depths)
N = len(reaction_depths)
tau_c_line = np.logspace(np.log10(tau_c.min()), np.log10(tau_c.max()), 200)
tau_r_boundary = (d - 2) * tau_c_line * N / sum_rd
# Clip to axis range
mask = (tau_r_boundary >= tau_r.min()) & (tau_r_boundary <= tau_r.max())
ax.plot(tau_c_line[mask], tau_r_boundary[mask], 'r-', linewidth=2, label='No stalling boundary')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=True)
ax.set_xlabel(r'$\tau_c$ (µs)')
ax.set_ylabel(r'$\tau_r$ (µs)')
# ax.set_title('Stalling time vs $\\tau_c$ and $\\tau_r$')
cbar = plt.colorbar(pc, ax=ax, label='Stalling time (µs)')
def fmt_sci(x, pos):
    if x == 0:
        return '0'
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / 10**exp
    if abs(mant - round(mant, 1)) < 1e-9:
        mant = int(round(mant))
        return rf'${mant} \times 10^{{{exp}}}$'
    return rf'${mant:.1f} \times 10^{{{exp}}}$'
cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_sci))
plt.tight_layout()
plt.savefig('reaction_limited_heatmap.pdf', bbox_inches='tight', pad_inches=0.1)
print('Saved reaction_limited_heatmap.pdf')
plt.show()
