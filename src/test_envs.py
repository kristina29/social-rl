from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from citylearn.agents.sac import SAC

from citylearn.base import Environment
from gym import Env, spaces


class TestEnv1(Environment, Env):
    def __init__(self, simulation_end_time_step, **kwargs):
        r"""Initialize `TestEnv1`.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super classes.
        """
        self.simulation_end_time_step = simulation_end_time_step
        self.observation_names = [['daylight_savings_status']]
        self.observation_space = [spaces.Box(low=np.array([0]), high=np.array([0]), dtype=np.float32)]
        self.action_space = [spaces.Box(low=np.array([0]), high=np.array([0]), dtype=np.float32)]
        super().__init__(**kwargs)

    def get_building_information(self):
        return []

    @property
    def done(self) -> bool:
        """Check if simulation has reached completion."""
        return self.time_step == self.time_steps - 1

    @property
    def time_steps(self) -> int:
        """Number of simulation time steps."""

        return (self.simulation_end_time_step - self.simulation_start_time_step) + 1

    @property
    def simulation_start_time_step(self) -> int:
        """Time step to end reading from data files."""

        return 0

    @property
    def simulation_end_time_step(self) -> int:
        """Time step to end reading from data files."""

        return self.__simulation_end_time_step

    @simulation_end_time_step.setter
    def simulation_end_time_step(self, simulation_end_time_step: int):
        assert simulation_end_time_step >= 0, 'simulation_end_time_step must be >= 0'
        self.__simulation_end_time_step = simulation_end_time_step

    def step(self, action: float) -> Tuple[None, float, bool, None]:
        """Apply action and advance to next time step.

        Returns
        -------
        observations: None
            :attr:`observations` current value.
        reward: float
            :meth:`get_reward` current value.
        done: bool
            A boolean value for if the episode has ended, in which case further :meth:`step` calls will return undefined results.
            A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
            a certain timelimit was exceeded, or the physics simulation has entered an invalid observation.
        """

        self.next_time_step()
        reward = [1]

        return [0], reward, self.done, None

    def next_time_step(self):
        r"""Advance to next `time_step`."""
        super().next_time_step()

    def reset(self):
        r"""Reset `Env` to initial state.
        """

        # object reset
        super().reset()

        return [0]


if __name__ == '__main__':
    env = TestEnv1(simulation_end_time_step=1)
    agent = SAC(env=env, seed=2, start_training_time_step=0, end_exploration_time_step=0, batch_size=1, alpha=0)
    losses, rewards = agent.learn(episodes=100, deterministic_finish=True)

    fig, ax = plt.subplots()
    ax.plot(rewards)
    ax.set_title('Rewards')

    fig, ax = plt.subplots()
    ax.plot(losses['q1_losses'], label='q1_losses')
    ax.plot(losses['q2_losses'], label='q2_losses')
    ax.plot(losses['policy_losses'], label='policy_losses')
    ax.legend()
    ax.set_title('Losses')

    plt.show()
