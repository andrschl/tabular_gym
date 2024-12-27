"""
Test transferability to new environments.
"""

# ----------------------------------------------------------------------------------------------------------------------
# imports
import sys
sys.path.append('../')
import numpy as np
import random
from utils.geometric_tools import *
from env.windy_gridworld import *
from algs.cmdp import cmdp_gda, regularization
from algs.cirl import reward
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
seed = 0
wind_level = 1.0
N = 1000000
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

# ----------------------------------------------------------------------------------------------------------------------
# Fix random seeds for reproducibility
random.seed(seed)
np.random.seed(seed)

# ----------------------------------------------------------------------------------------------------------------------
# create environment
params = {
    'grid_height': 6,
    'grid_width': 6,
    'wind_level': float(1.0),
    'gamma': 0.9
}
n = params['grid_height'] * params['grid_width']
m = 4
nu0 = np.ones(n) / n
from scipy.linalg import subspace_angles
env0 = WindyGridworld(6, 6, 0.0, 0.9, nu0=nu0)
env1 = WindyGridworld(6, 6, 0.0, 0.9, nu0=nu0)
env1.P = np.roll(env1.P, 1, axis=1)  # apply cyclic shift to actions
envW = WindyGridworld(**params, nu0=nu0, wind_direction='W')
envS = WindyGridworld(**params, nu0=nu0, wind_direction='S')
envSW = WindyGridworld(**params, nu0=nu0, P=0.5*envW.P + 0.5*envS.P)
envs = [envW, envSW]
print('2nd principal angle: ', subspace_angles(env0.A_matrix(), envSW.A_matrix())[-2])
Phi = rearrange(np.eye(n*m), '(s a) k -> s a k', s = n, a = m)
print('params', params)
print('wind_level', wind_level)
print('seed', seed)

save_path = 'data/multi_expert_irl_sparse/beta' + str(beta) + '/wind_' + str(wind_level) +  '/'

# Evaluate rewards
e_constant = unit_vector(np.ones((n*m,1)), ord=2)
eta_p = (1-envW.gamma) /beta * 1
eta_xi = 1.0

JSW_E = np.zeros((4, 10))
JSW_hat = np.zeros((4, 10))

JS_E = np.zeros((4, 10))
JS_hat = np.zeros((4, 10))

Jrolled_E = np.zeros((4, 10))
Jrolled_hat = np.zeros((4, 10))

rdist = np.zeros((4, 10))
for seed in range(0,10):
    for i, N in enumerate([1000, 10000, 100000, 1000000]):
        load_path = 'data/multi_expert_irl_sparse/beta' + str(beta) + '/wind_' + str(wind_level) + '/seed_' + str(seed) + '/' + str(N) + '_rollouts/'
        wE = pd.read_csv(load_path + 'wE.csv').to_numpy()[:, 0]
        what = pd.read_csv(load_path + 'w.csv').to_numpy()[:, 0]
        rE = reward(wE, Phi)
        rhat = reward(what, Phi)
        rdist[i, seed] = quotient_norm(wE-what, e_constant, is_orth=True)
        print('-----------')
        print('GDA solution')

        # South-west wind
        envSW.r = rE
        piSW_E, _, _ = cmdp_gda(envSW, beta, eta_p, eta_xi, max_iters=2e3, tol=-1, mode='alt_gda', r=rE,
                               n_v_tot_eval_steps=50, n_v_c_eval_steps=50, logging=True, check_steps=500)
        occSW_E = envSW.policy2stateactionocc(piSW_E)
        envSW.r = rhat
        piSW_hat, _, _ = cmdp_gda(envSW, beta, eta_p, eta_xi, max_iters=2e3, tol=-1, mode='alt_gda', r=rhat,
                                 n_v_tot_eval_steps=50, n_v_c_eval_steps=50, logging=True, check_steps=500)
        occSW_hat = envSW.policy2stateactionocc(piSW_hat)
        JSW_E[i, seed] = np.sum(occSW_E * rE / (1 - envW.gamma)) - regularization(envSW, occSW_E, beta)
        JSW_hat[i, seed] = np.sum(occSW_hat * rE / (1 - envW.gamma)) - regularization(envSW, occSW_hat, beta)
        fig0 = gv.plot_gridworld(envSW, piSW_E, values=rE[:, 0])
        fig0.tight_layout()
        fig0.savefig(save_path + 'seed_' + str(seed) + '/' + str(N) + '_rollouts/piSW_E.png')
        plt.close(fig0)
        fig1 = gv.plot_gridworld(envSW, piSW_hat, values=rhat[:, 0])
        fig1.tight_layout()
        fig1.savefig(save_path + 'seed_' + str(seed) + '/' + str(N) + '_rollouts/piSW_hat.png')
        plt.close(fig1)

        # South wind
        envS.r = rE
        piS_E, _, _ = cmdp_gda(envS, beta, eta_p, eta_xi, max_iters=2e3, tol=-1, mode='alt_gda', r=rE,
                                n_v_tot_eval_steps=50, n_v_c_eval_steps=50, logging=True, check_steps=500)
        occS_E = envS.policy2stateactionocc(piS_E)
        envS.r = rhat
        piS_hat, _, _ = cmdp_gda(envS, beta, eta_p, eta_xi, max_iters=2e3, tol=-1, mode='alt_gda', r=rhat,
                                  n_v_tot_eval_steps=50, n_v_c_eval_steps=50, logging=True, check_steps=500)
        occS_hat = envS.policy2stateactionocc(piS_hat)
        JS_E[i, seed] = np.sum(occS_E * rE / (1 - envW.gamma)) - regularization(envS, occS_E, beta)
        JS_hat[i, seed] = np.sum(occS_hat * rE / (1 - envW.gamma)) - regularization(envS, occS_hat, beta)
        fig0 = gv.plot_gridworld(envS, piS_E, values=rE[:, 0])
        fig0.tight_layout()
        fig0.savefig(save_path + 'seed_' + str(seed) + '/' + str(N) + '_rollouts/piS_E.png')
        plt.close(fig0)
        fig1 = gv.plot_gridworld(envS, piS_hat, values=rhat[:, 0])
        fig1.tight_layout()
        fig1.savefig(save_path + 'seed_' + str(seed) + '/' + str(N) + '_rollouts/piS_hat.png')
        plt.close(fig1)

        # Roll actions
        env1.r = rE
        pi1_E, _, _ = cmdp_gda(env1, beta, eta_p, eta_xi, max_iters=2e3, tol=-1, mode='alt_gda', r=rE,
                                n_v_tot_eval_steps=50, n_v_c_eval_steps=50, logging=True, check_steps=500)
        occ1_E = env1.policy2stateactionocc(pi1_E)
        env1.r = rhat
        pi1_hat, _, _ = cmdp_gda(env1, beta, eta_p, eta_xi, max_iters=2e3, tol=-1, mode='alt_gda', r=rhat,
                                  n_v_tot_eval_steps=50, n_v_c_eval_steps=50, logging=True, check_steps=500)
        occ1_hat = env1.policy2stateactionocc(pi1_hat)
        Jrolled_E[i, seed] = np.sum(occ1_E * rE / (1 - env1.gamma)) - regularization(env1, occ1_E, beta)
        Jrolled_hat[i, seed] = np.sum(occ1_hat * rE / (1 - env1.gamma)) - regularization(env1, occ1_hat, beta)
        fig0 = gv.plot_gridworld(env1, pi1_E, values=rE[:, 0])
        fig0.tight_layout()
        fig0.savefig(save_path + 'seed_' + str(seed) + '/' + str(N) + '_rollouts/pirolled_E.png')
        plt.close(fig0)
        fig1 = gv.plot_gridworld(env1, pi1_hat, values=rhat[:, 0])
        fig1.tight_layout()
        fig1.savefig(save_path + 'seed_' + str(seed) + '/' + str(N) + '_rollouts/pirolled_hat.png')
        plt.close(fig1)

lSW = JSW_E - JSW_hat
pd.DataFrame(JSW_E.T, columns=['Ne3', 'Ne4', 'Ne5', 'Ne6']).to_csv(save_path + "JSW_E.csv", index=False)
pd.DataFrame(JSW_hat.T, columns=['Ne3', 'Ne4', 'Ne5', 'Ne6']).to_csv(save_path + "JSW_hat.csv", index=False)
pd.DataFrame(lSW.T, columns=['Ne3', 'Ne4', 'Ne5', 'Ne6']).to_csv(save_path + "lSW.csv", index=False)
lS = JS_E - JS_hat
pd.DataFrame(JS_E.T, columns=['Ne3', 'Ne4', 'Ne5', 'Ne6']).to_csv(save_path + "JS_E.csv", index=False)
pd.DataFrame(JS_hat.T, columns=['Ne3', 'Ne4', 'Ne5', 'Ne6']).to_csv(save_path + "JS_hat.csv", index=False)
pd.DataFrame(lS.T, columns=['Ne3', 'Ne4', 'Ne5', 'Ne6']).to_csv(save_path + "lS.csv", index=False)
lrolled = Jrolled_E - Jrolled_hat
pd.DataFrame(Jrolled_E.T, columns=['Ne3', 'Ne4', 'Ne5', 'Ne6']).to_csv(save_path + "Jrolled_E.csv", index=False)
pd.DataFrame(Jrolled_hat.T, columns=['Ne3', 'Ne4', 'Ne5', 'Ne6']).to_csv(save_path + "Jrolled_hat.csv", index=False)
pd.DataFrame(lrolled.T, columns=['Ne3', 'Ne4', 'Ne5', 'Ne6']).to_csv(save_path + "lrolled.csv", index=False)
pd.DataFrame(rdist.T, columns=['Ne3', 'Ne4', 'Ne5', 'Ne6']).to_csv(save_path + "rdist.csv", index=False)