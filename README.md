# Transferability in regularized IRL
This code implements the experiments of the paper Towards the Transferability of Rewards Recovered via Regularized Inverse Reinforcement Learning.

## Experiments
- To generate synthetic expert data run `experiments/run_get_windy_experts.sh`
- To run IRL run `experiments/run_multi_expert_irl.sh`
- To evaluate the transferability run `experiments/run_check_transferability.sh`
- For generating the plot run `experiments/plotting.py`
- The computations for Example 3.2 are done in the notebook `experiments/example.ipynb`