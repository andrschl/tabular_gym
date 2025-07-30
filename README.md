# Tabular Gym

[**Installation**](#installation) |
[**Quick Start**](#quick-start) |
[**Examples**](#examples) |
[**Contributing**](#contributing)

A tabular reinforcement learning gym for constrained MDPs and inverse reinforcement learning. This package implements the experiments of the paper [Towards the Transferability of Rewards Recovered via Regularized Inverse Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2628d4d3b054c2d7ad33ab03435204f4-Abstract-Conference.html).

## Installation

```bash
git clone <repository-url>
cd tabular_gym
pip install -e .
```

## Quick Start

```python
import numpy as np
from tabular_gym import Gridworld, WindyGridworld
from tabular_gym.algs import irl_gda, empirical_expert_feature_expectation

# Create a gridworld environment
env = Gridworld(grid_width=5, grid_height=5, noise=0.1, gamma=0.9)

# Create a windy gridworld for transfer experiments
windy_env = WindyGridworld(grid_width=5, grid_height=5, wind_level=0.5, gamma=0.9)

# Run inverse reinforcement learning
# ... (see examples for full usage)
```

## Examples

The `experiments/` directory contains example scripts that demonstrate how to use the package:

- To generate synthetic expert data run `experiments/run_get_windy_experts.sh`
- To run IRL run `experiments/run_multi_expert_irl.sh`
- To evaluate the transferability run `experiments/run_check_transferability.sh`
- For generating the plot run `experiments/plotting.py`
- The computations for Example 3.2 are done in the notebook `experiments/example.ipynb`


## Contributing
If you would like to contribute to the project please reach out to [Andreas Schlaginhaufen](mailto:andreas.schlaginhaufen@epfl.ch?subject=[transfer_irl]%20Contribution%20to%20transfer_irl). If you found this library useful in your research, please consider citing the following paper:
```
@inproceedings{NEURIPS2024_2628d4d3,
 author = {Schlaginhaufen, Andreas and Kamgarpour, Maryam},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {21461--21501},
 publisher = {Curran Associates, Inc.},
 title = {Towards the Transferability of Rewards Recovered via Regularized Inverse Reinforcement Learning},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/2628d4d3b054c2d7ad33ab03435204f4-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```
