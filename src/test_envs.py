from abc import ABC
from typing import List, Tuple

import numpy as np
from gym import Env, spaces
from matplotlib import pyplot as plt

from citylearn.base import Environment

from citylearn.agents.sac import SAC


class TestEnv(Environment, Env):
    def __init__(self, **kwargs):
        self.__rewards = []
        self.__time_step = 0
        self.observations = [0]
        self.observation_names = [['bla']]
        self.action_space = [spaces.Box(low=np.array([0.]), high=np.array([5.]), dtype=np.float32)]
        self.observation_space = [spaces.Box(low=np.array(0, dtype='float32'), high=np.array(5, dtype='float32'))]
        self.simulation_end_time_step = 1000
        self.simulation_start_time_step = 1

    def step(self, actions: List[List[float]]) -> Tuple[List[List[float]], List[float], bool, dict]:
        if actions[0][0] > 2:
            reward = [100]
        else:
            reward = [-100]
        self.__rewards.append(reward)
        self.next_time_step()
        return self.observations, reward, self.done, None

    def get_building_information(self):
        return None

    def reset(self):
        return self.observations

    @property
    def time_step(self) -> int:
        r"""Current environment time step."""

        return self.__time_step

    def next_time_step(self):
        r"""Advance to next `time_step` value.

        Notes
        -----
        Override in subclass for custom implementation when advancing to next `time_step`.
        """

        self.__time_step += 1

    def reset_time_step(self):
        r"""Reset `time_step` to initial state.

        Sets `time_step` to 0.
        """

        self.__time_step = 0

    @property
    def time_steps(self) -> int:
        """Number of simulation time steps."""

        return (self.simulation_end_time_step - self.simulation_start_time_step) + 1

    @property
    def done(self) -> bool:
        """Check if simulation has reached completion."""
        return self.time_step == self.time_steps - 1


if __name__ == '__main__':

    env = TestEnv()
    agent = SAC(env=env, autotune_entropy=True, start_training_time_step=1, end_exploration_time_step=700)

    losses, rewards = agent.learn(episodes=2)

    rewards = rewards[0]
    #rewards = [value for sublist in rewards for value in sublist]
    rewards = [value for value in rewards]
    plt.plot(rewards)
    plt.show()