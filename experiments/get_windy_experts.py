"""
Generating the experts.
"""
# ----------------------------------------------------------------------------------------------------------------------
# imports
import numpy as np
from tabular_gym.env.windy_gridworld import WindyGridworld
from tabular_gym.algs.cmdp import cmdp_gda, regularization
from tabular_gym.algs.cirl import batched_empirical_feature_expectation
import tabular_gym.visualization.gridworld_vis as gv
import matplotlib
from einops import rearrange, repeat
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import copy
import random
import numpy.random as rn
import pandas as pd
import argparse
from pathlib import Path
from scipy.stats import entropy
float_formatter = "{:.8f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

# ----------------------------------------------------------------------------------------------------------------------
# Parse command line arguments
# default values
N = 1000
seed = 1
wind_level = 0.5
wind_directions = ['N', 'E']
beta = 0.3

# parse arguments
parser = argparse.ArgumentParser(description = "Hyperparameters to pass")
parser.add_argument('--wind_level')
parser.add_argument('--N')
parser.add_argument('--seed')
parser.add_argument('--beta')
args = parser.parse_args()
if args.wind_level:
    wind_level = float(args.wind_level)
if args.beta:
    beta = float(args.beta)
if args.N:
    N = int(args.N)
if args.seed:
    seed = int(args.seed)

data_path = 'data/expert_data_sparse/beta' + str(beta) + '/wind_'+str(wind_level)+'/seed_'+str(seed)+'/'
Path(data_path).mkdir(parents=True, exist_ok=True)
plotting = True

# ----------------------------------------------------------------------------------------------------------------------
# Fix random seeds for reproducibility
print('args: ', args)
random.seed(seed)
np.random.seed(seed)

# ----------------------------------------------------------------------------------------------------------------------
# create environment
params = {
    'grid_height': 6,
    'grid_width' : 6,
    # 'wind_level': float(wind_level),
    'gamma': 0.9
}
n = params['grid_height'] * params['grid_width']
m = 4
nu0 = np.ones(n) / n
rE = np.zeros(n*m)
rE[rn.choice(n*m, size=10, replace=False)] = rn.choice([-1, 1], size=10)
rE = rearrange(rE, '(s a) -> s a', a=m)
print(rE)
env0 = WindyGridworld(**params, nu0=nu0, r=rE, wind_direction=wind_directions[0], wind_level=wind_level)
env1 = WindyGridworld(**params, nu0=nu0, r=rE, wind_direction=wind_directions[1], wind_level=wind_level)

print('params', params)
print('N', N)
print('wind_level', wind_level)
print('seed', seed)

# ----------------------------------------------------------------------------------------------------------------------
# Get expert
# LP solution (as sanity check)
print('-----------')
# print('LP solution')
# occ_lp, sol = env.lp_solve()
# print(sol.message)
# print('objective: ', -sol.fun)
# policy_lp = env.occ2policy(occ_lp)

# Approximate solution to regularized problem
eta_p = (1-env0.gamma) /beta * 1
eta_xi = 1.0
print('-----------')
print('GDA solution')
piE0, _, _ = cmdp_gda(env0, beta , eta_p, eta_xi, max_iters=2e4, tol=-1, mode='alt_gda',
                                    n_v_tot_eval_steps=50, n_v_c_eval_steps=50, logging=True, check_steps=1000)
occE0 = env0.policy2stateactionocc(piE0)
print('unreg objective: ', np.sum(occE0 * env0.r / (1-env0.gamma)))
print('reg objective: ', np.sum(occE0 * env0.r / (1-env0.gamma)) - regularization(env0, occE0, beta))
piE1, _, _ = cmdp_gda(env1, beta , eta_p, eta_xi, max_iters=2e4, tol=-1, mode='alt_gda',
                                    n_v_tot_eval_steps=50, n_v_c_eval_steps=50, logging=True, check_steps=1000)
occE1 = env1.policy2stateactionocc(piE1)
print('unreg objective: ', np.sum(occE1 * env1.r / (1-env1.gamma)))
print('reg objective: ', np.sum(occE1 * env1.r / (1-env1.gamma)) - regularization(env1, occE1, beta))

# ----------------------------------------------------------------------------------------------------------------------
# Get feature expectation
n_features = n
T = 100
# Phi= repeat(np.eye(env.n), 'x y -> x a y', a = env.m)
Phi = rearrange(np.eye(n*m), '(s a) k -> s a k', s=n, a=m)
for N in [int(1e3), int(1e4), int(1e5), int(1e6)]: #, 1e4, 1e5, 1e6]:
    random.seed(seed)
    np.random.seed(seed)
    sigmaE0 = batched_empirical_feature_expectation(env0, piE0, Phi, N, T)
    sigmaE1 = batched_empirical_feature_expectation(env1, piE1, Phi, N, T)
    save_path = data_path + str(N) + '_rollouts/'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sigmaE0, columns=['sigmaE']).to_csv(save_path + "sigmaE0.csv", index=False)
    pd.DataFrame(sigmaE1, columns=['sigmaE']).to_csv(save_path + "sigmaE1.csv", index=False)

sigmaE0true = np.einsum('jki,jk->i', Phi, occE0) / (1-env0.gamma)
sigmaE1true = np.einsum('jki,jk->i', Phi, occE1) / (1-env0.gamma)

# Store to csv
pd.DataFrame(sigmaE0true, columns=['sigmaE']).to_csv(data_path + "sigmaE0true.csv", index=False)
pd.DataFrame(sigmaE1true, columns=['sigmaE']).to_csv(data_path + "sigmaE1true.csv", index=False)

pd.DataFrame(piE0).to_csv(data_path + "piE0.csv", index=False)
pd.DataFrame(piE1).to_csv(data_path + "piE1.csv", index=False)

pd.DataFrame(occE0).to_csv(data_path + "muE0.csv", index=False)
pd.DataFrame(occE1).to_csv(data_path + "muE1.csv", index=False)

pd.DataFrame(rE).to_csv(data_path + "rE.csv", index=False)


# ----------------------------------------------------------------------------------------------------------------------
# Plotting
if plotting:

    print('--------')
    coloring = rE[:,0]
    fig0 = gv.plot_gridworld(env0, piE0, values=coloring)
    fig0.tight_layout()
    fig0.savefig(data_path + 'piE0.png')
    fig1 = gv.plot_gridworld(env1, piE1, values=coloring)
    fig1.tight_layout()
    fig1.savefig(data_path + 'piE1.png')
