"""
Learn rewards from two agents.
"""

# ----------------------------------------------------------------------------------------------------------------------
# imports
import sys
sys.path.append('../')
import numpy as np
import numpy.random as rn
import random
from utils.geometric_tools import *
from env.windy_gridworld import *
from algs.cirl import reward, multi_expert_irl_gda
import visualization.gridworld_vis as gv
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
from einops import rearrange, reduce, repeat, einsum
import pandas as pd
import argparse
from pathlib import Path
import wandb
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

# ----------------------------------------------------------------------------------------------------------------------
# Parse command line arguments
# default values
true_expert = False
seed = 1
wind_level = 1.0
wind_directions = ['N', 'E']
N = 1000000
projection = 'l2_ball'
# projection = 'l2_ball'
# projection = 'linf_ball'
# projection = None
beta = 0.3

# parse arguments
parser = argparse.ArgumentParser(description = 'Setting')
parser.add_argument('--wind_level')
parser.add_argument('--N')
parser.add_argument('--seed')
parser.add_argument('--beta')
args = parser.parse_args()

if args.wind_level:
    wind_level = float(args.wind_level)
if args.N:
    N = int(args.N)
if args.seed:
    seed = int(args.seed)
if args.beta:
    beta = float(args.beta)

# load experts
load_path = 'data/expert_data_sparse/beta' + str(beta) + '/wind_'+str(wind_level)+'/seed_'+str(seed)+'/'  # + str(N) + '_rollouts/'
save_path = 'data/multi_expert_irl_sparse/beta' + str(beta) + '/wind_' + str(wind_level) + '/seed_' + str(seed) + '/' + str(N) + '_rollouts/'
run_name = 'multi_expert_irl_wind_' + str(wind_level) + '_' + str(N) + '_rollouts'

Path(load_path).mkdir(parents=True, exist_ok=True)
Path(save_path).mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------------------------------------------------
# Fix random seeds for reproducibility
random.seed(seed)
np.random.seed(seed)

# ----------------------------------------------------------------------------------------------------------------------
# create environment
params = {
    'grid_height': 6,
    'grid_width': 6,
    'wind_level': float(wind_level),
    'gamma': 0.9
}
n = params['grid_height'] * params['grid_width']
m = 4
nu0 = np.ones(n) / n
rE = pd.read_csv(load_path + 'rE.csv').to_numpy()
env0 = WindyGridworld(**params, nu0=nu0, r=rE, wind_direction=wind_directions[0])
env1 = WindyGridworld(**params, nu0=nu0, r=rE, wind_direction=wind_directions[1])
envs = [env0, env1]

# ----------------------------------------------------------------------------------------------------------------------
# Hyperparams
run_config = {
    'eta_p': (1-env0.gamma) /beta * 1.0,
    'eta_xi': 1e1,
    # 'eta_w': 5e-2,
    'eta_w': 5e-2,
    # 'eta_w': 2e-2,

    #  'eta_w': 2e-1,
    # 'eta_w': 5e-1,
    'lambda_w': 0.00
}

wandb_logging = False
wandb_sweep = False
logging = True

if wandb_logging:
    wandb.init(project=run_name, config=run_config)

print(args)
print('params ', params)
print('wind_level', wind_level)
print('seed', seed)
print('NE ', N)

# ----------------------------------------------------------------------------------------------------------------------
# Initialize reward model
Phi = rearrange(np.eye(n*m), '(s a) k -> s a k', s = n, a = m)
wE = rearrange(rE, 's a -> (s a)')
radius = 1000 # set this big enough, so we don't always hit the boundary.
print('rE norm: ', norm(rE, ord=2))

# Load expert data
sigmaEs = []
sigmaEtrues = []
piEs = []
muEs = []
for k, env in enumerate(envs):
    sigmaEtrues.append(pd.read_csv(load_path + 'sigmaE{}true.csv'.format(k)).to_numpy()[:, 0])
    if true_expert:
        sigmaEs = sigmaEtrues
    else:
        sigmaEs.append(pd.read_csv(load_path + str(N) + '_rollouts/sigmaE{}.csv'.format(k)).to_numpy()[:, 0])

    # print optimal policy mismatch for debugging
    piEs.append(pd.read_csv(load_path + 'piE{}.csv'.format(k)).to_numpy())
    muEs.append(pd.read_csv(load_path + 'muE{}.csv'.format(k)).to_numpy())
    v_f = env.approx_vector_cost_eval(np.zeros((env.n, Phi.shape[2])), Phi, piEs[k], max_iters = 1e3, tol = 1e-5, logging = False)
    w_grad = np.einsum('i,ij->j', env.nu0, v_f) - sigmaEs[k]
    print('feature expectation estimation error ', norm(sigmaEtrues[k]-sigmaEs[k], ord=np.inf))
    print('optimal feature expectation mismatch ', np.max(np.abs(w_grad)), np.max(np.abs(np.einsum('sad,sa->d', Phi, muEs[k]) / (1-env.gamma) - sigmaEs[k])))
print('sum of feature expectation estimation errors ', norm(sum(sigmaEtrues)-sum(sigmaEs), ord=np.inf))

# ----------------------------------------------------------------------------------------------------------------------
# Training
policies_irl, values_irl, w = multi_expert_irl_gda(envs, beta, run_config['eta_p'], run_config['eta_w'], Phi,
            sigmaEs, n_p_steps=1, max_iters=3e4, mode='alt_gda', n_v_tot_eval_steps=50,
            n_v_f_eval_steps=50, check_steps=1000, wandb_log=wandb_logging, projection=projection,
            radius=radius, logging=logging, mu_trues=muEs, w_true=wE, N=100, T=100, lambda_=0.0, w0=rn.normal(size=wE.shape, scale=1))

# -----------------------------------------------------------------------------------------------------------------------
# Store to csv
for i, env in enumerate(envs):
    pd.DataFrame(policies_irl[i]).to_csv(save_path + "policy{}.csv".format(i), index=False)
    pd.DataFrame(env.policy2stateactionocc(policies_irl[i])).to_csv(save_path + "mu{}.csv".format(i), index=False)
pd.DataFrame(w).to_csv(save_path + "w.csv", index=False)
pd.DataFrame(wE).to_csv(save_path + "wE.csv", index=False)

# ----------------------------------------------------------------------------------------------------------------------
# Illustration
for k, env in enumerate(envs):
    fig = gv.plot_gridworld(env, policies_irl[k], values=reward(w, Phi)[:,0])
    fig.tight_layout()
    fig.savefig(save_path + 'imitation{}.png'.format(k))
