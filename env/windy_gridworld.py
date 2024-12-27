from env.gridworld import Gridworld
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Union, Callable

Point = np.ndarray  # type for state and action coordinates, np.array([int, int])

class WindyGridworld(Gridworld):
    """
    Stochastic version of a WindyGridworld CMDP.

    additional Attributes:
        wind_level: Float between 0 and 1.
        wind direction: String in 'east', 'north', 'west', 'south'
    """

    def __init__(self, grid_width: int, grid_height: int, wind_level: float, gamma: float,
                 nu0: Optional[np.ndarray] = None, r: Optional[np.ndarray] = None, wind_direction='N',
                 constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None, P=None) -> None:

        # gridworld specific attributes
        self.grid_height: int = grid_height
        self.grid_width: int = grid_width
        self.wind_level = wind_level

        # general CMDP attributes
        n = grid_width * grid_height
        m = 4
        noise = 0.0
        super().__init__(grid_width, grid_height, noise, gamma, nu0=nu0, r=r, constraints=constraints, P=np.zeros((n, m, n)))
        self.action_dict = {'E': self.actions[0], 'N': self.actions[1], 'W': self.actions[2], 'S': self.actions[3]}
        self.P: np.ndarray = np.array(
            [[[(1-self.wind_level) * self._transition_dynamics(s, a, s_next) + self.wind_level * self._wind_dynamics(s, a, s_next, direction=wind_direction)
               for s_next in range(self.n)]
                for a in range(self.m)]
                 for s in range(self.n)]) if P is None else P

    def _wind_dynamics(self, s: int, a: int, s_next: int, direction='N') -> float:
        """
        Get the probability of transitioning from state s to state s_next given
        action a in the windy dynamics.

        :param s: State int.
        :param a: Action int.
        :param s_next: State int.
        :param direction: 'east', 'north', 'west', 'south'
        :return: P(s_next | s, a)
        """

        s_next = self.int2point(s_next)
        s = self.int2point(s)
        a = self.actions[a]
        wind = self.action_dict[direction]
        s_next_wind = s + a + wind
        s_next_wind = np.array([np.clip(s_next_wind[0], 0, self.grid_width-1), np.clip(s_next_wind[1], 0, self.grid_height-1)])

        if (s_next_wind == s_next).all():
            return 1.0
        else:
            return 0.0