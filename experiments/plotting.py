"""
Plotting for finite sample setting.
"""

import numpy as np
import pandas as pd
from tabular_gym.utils.geometric_tools import *
from scipy.linalg import svdvals, subspace_angles
from tabular_gym.env.windy_gridworld import *
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("TkAgg")
font = {'family' : 'normal',
        'weight': 'black',
        'size'   : 20}
mpl.rc('font', **font)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})
plt.rcParams['text.usetex'] = True
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r"\usepackage{amsmath}"
                              r"\usepackage{bm}")
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

def quantile_plot(ax, data, label):

    # Calculation of medians and quantiles
    medians = [np.median(group, axis=0) for group in data.T]
    lower_quantiles = [np.percentile(group, 20, axis=0) for group in data.T]
    upper_quantiles = [np.percentile(group, 80, axis=0) for group in data.T]

    # Calculate the error bars (distance from median)
    lower_errors = [median - lq for median, lq in zip(medians, lower_quantiles)]
    upper_errors = [uq - median for median, uq in zip(medians, upper_quantiles)]
    asymmetric_error = [lower_errors, upper_errors]

    # X-axis positions
    x_positions = [1, 2, 3, 4]

    # Creating the plot
    ax.errorbar(x_positions, medians, yerr=asymmetric_error, fmt='o', capsize=3,
                label=label, alpha = .75, linestyle = ':', capthick = 1)
    data = {
        'x': [1, 2, 3, 4],
        'y1': lower_quantiles,
        'y2': upper_quantiles}
    ax.fill_between(**data, alpha=.25)
    return ax

beta = 0.3
load_path = 'data/multi_expert_irl_sparse/beta' + str(beta)

# plot
fig, (ax3, ax1, ax0, ax2) = plt.subplots(1,4, figsize=(15,5))
#
# fig1, ax1 = plt.subplots(figsize=(8, 6))
# fig2, ax2 = plt.subplots(figsize=(8, 6))
for wind_level in [0.01, 0.1, 0.5, 1.0]:
    # import data
    rdist = pd.read_csv(load_path + '/wind_' + str(wind_level) + '/rdist.csv').to_numpy()
    lrolled = pd.read_csv(load_path + '/wind_' + str(wind_level) + '/lrolled.csv').to_numpy()
    lS = pd.read_csv(load_path + '/wind_' + str(wind_level) + '/lS.csv').to_numpy()
    ax1 = quantile_plot(ax1, rdist, label=r'$\beta={}$'.format(wind_level))
    ax2 = quantile_plot(ax2, lrolled, label=r'$\beta={}$'.format(wind_level))
    ax0 = quantile_plot(ax0, lS, label=r'$\beta={}$'.format(wind_level))
    # # plotting
    # rdist_med = np.median(rdist, axis=0)
    # rdist_quantiles = np.quantile(rdist, [0.1, 0.9], axis=0)
    # lSW_med = np.median(lSW, axis=0)
    # lSW_quantiles = np.quantiles(lSW, [0.1, 0.9], axis=0)
    # lrolled_med = np.median(lrolled, axis=0)
    # lrolled_quantiles = np.quantiles(lrolled, [0.1, 0.9], axis=0)

# Adding labels and title
ax1.set_xticks([1, 2, 3, 4])
ax1.set_xticklabels([r'$10^3$', r'$10^4$', r'$10^5$', r'$10^6$'])
ax1.set_xlabel(r'$N^E$')
ax1.set_ylabel('reward distance')
ax2.set_xticks([1, 2, 3, 4])
ax2.set_xticklabels([r'$10^3$', r'$10^4$', r'$10^5$', r'$10^6$'])
ax2.set_xlabel(r'$N^E$')
ax2.set_ylabel(r'transferability')
ax0.set_xticks([1, 2, 3, 4])
ax0.set_xticklabels([r'$10^3$', r'$10^4$', r'$10^5$', r'$10^6$'])
ax0.set_xlabel(r'$N^E$')
ax0.set_ylabel(r'transferability')

# Adding a legend
ax1.legend()

# Adding grid
ax0.grid(True)
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)

# Plot principal angles
N = 50
width = 6
height = 6
n = width * height
nu0 = unit_vector(np.ones(n), ord=1)
gamma = 0.9
A_angles = np.zeros(N)
levels = np.zeros(N)
sing_val_diffs = np.zeros(N)
for i, level in enumerate(np.linspace(0, 1, N)):
    levels[i] = level
    envN = WindyGridworld(width, height, level, gamma, nu0=nu0, wind_direction='N')
    envE = WindyGridworld(width, height, level, gamma, nu0=nu0, wind_direction='E')
    A1 = envN.A_matrix()
    A2 = envE.A_matrix()
    A_angles[i] = subspace_angles(A1, A2)[-2]

ax3.plot(levels, A_angles,  linestyle='-', linewidth=3)
ax3.set_xlim([0, None])
ax3.set_ylim([0, None])
ax3.set_xlabel(r'$\beta$')
ax3.set_ylabel(r'$\theta_2({P^0}, {P^1})$')
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
fig.savefig('figures/experiments.png', dpi=400)
# fig1.savefig('figures/reward_distances.png', dpi=400)
# fig2.savefig('figures/transferability.png', dpi=400)





# Show the plot
plt.show()
