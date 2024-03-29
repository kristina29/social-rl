import copy
import inspect
import logging
import os
from pathlib import Path
import pickle
from typing import Any, List, Mapping, Union

from gym import spaces

from citylearn.base import Environment
from citylearn.citylearn import CityLearnEnv


LOGGER = logging.getLogger()
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.pyplot').disabled = True


class Agent(Environment):
    def __init__(self, env: CityLearnEnv, **kwargs):
        r"""Initialize `Agent`.

        Parameters
        ----------
        env : CityLearnEnv
            CityLearn environment.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """

        self.env = env
        self.observation_names = self.env.observation_names
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.building_information = self.env.get_building_information()

        arg_spec = inspect.getfullargspec(super().__init__)
        kwargs = {
            key: value for (key, value) in kwargs.items()
            if (key in arg_spec.args or (arg_spec.varkw is not None))

        }
        super().__init__(**kwargs)

    @property
    def observation_names(self) -> List[List[str]]:
        """Names of active observations that can be used to map observation values."""

        return self.__observation_names

    @property
    def observation_space(self) -> List[spaces.Box]:
        """Format of valid observations."""

        return self.__observation_space

    @property
    def action_space(self) -> List[spaces.Box]:
        """Format of valid actions."""

        return self.__action_space

    @property
    def building_information(self) -> List[Mapping[str, Any]]:
        """Building metadata."""

        return self.__building_information

    @property
    def action_dimension(self) -> List[int]:
        """Number of returned actions."""
        return [s.shape[0] for s in self.action_space]

    @property
    def actions(self) -> List[List[List[Any]]]:
        """Action history/time series."""

        return self.__actions

    @observation_names.setter
    def observation_names(self, observation_names: List[List[str]]):
        self.__observation_names = observation_names

    @observation_space.setter
    def observation_space(self, observation_space: List[spaces.Box]):
        self.__observation_space = observation_space

    @action_space.setter
    def action_space(self, action_space: List[spaces.Box]):
        self.__action_space = action_space

    @building_information.setter
    def building_information(self, building_information: List[Mapping[str, Any]]):
        self.__building_information = building_information

    @actions.setter
    def actions(self, actions: List[List[Any]]):
        for i in range(len(self.action_space)):
            self.__actions[i][self.time_step] = actions[i]

    def learn(
            self, episodes: int = None, keep_env_history: bool = None, env_history_directory: Union[str, Path] = None,
            deterministic: bool = None, deterministic_finish: bool = None, logging_level: int = None
    ) -> (Mapping[int, Mapping[str, List[float]]], List[float]):
        """Train agent.

        Parameters
        ----------
        episodes: int, default: 1
            Number of training episode greater :math:`\ge 1`.
        keep_env_history: bool, default: False
            Indicator to store environment state at the end of each episode.
        env_history_directory: Union[str, Path], optional
            Directory to save environment history to.
        deterministic: bool, default: False
            Indicator to take deterministic actions i.e. strictly exploit the learned policy.
        deterministic_finish: bool, default: False
            Indicator to take deterministic actions in the final episode.
        logging_level: int, default: 30
            Logging level where increasing the number silences lower level information.

        Return values
        ----------
        losses: Mapping[int, Mapping[str, List[float]]]
            Mapping of neural-network index to Mapping of neural-network name to loss values of training steps.
        rewards: List[List[float]]
            Reward value per training step.
        """

        episodes = 1 if episodes is None else episodes
        keep_env_history = False if keep_env_history is None else keep_env_history
        deterministic_finish = False if deterministic_finish is None else deterministic_finish
        deterministic = False if deterministic is None else deterministic
        self.__set_logger(logging_level)

        if keep_env_history:
            env_history_directory = Path(
                f'citylearn_learning_{self.env.uid}') if env_history_directory is None else env_history_directory
            os.makedirs(env_history_directory, exist_ok=True)

        else:
            pass

        losses = None
        rewards = []
        eval_results = {'1 - average_daily_renewable_share': [],
                        '1 - average_daily_renewable_share_grid': [],
                        '1 - used_pv_of_total_share': [],
                        'fossil_energy_consumption': []}
        kpi_min = 100
        best_state = None

        for episode in range(episodes):
            deterministic = deterministic or (deterministic_finish and episode >= episodes - 1)
            observations = self.env.reset()

            while not self.env.done:
                actions = self.predict(observations, deterministic=deterministic)

                # apply actions to citylearn_env
                next_observations, new_rewards, _, _ = self.env.step(actions)

                for i, r in enumerate(new_rewards):
                    if len(rewards) < i + 1:
                        rewards.append([])
                    rewards[i].append(r)

                # update
                if not deterministic:
                    new_losses = self.update(observations, actions, new_rewards, next_observations, done=self.env.done)

                    if losses is None:
                        losses = new_losses
                    else:
                        for i, losses_i in new_losses.items():
                            for name, value in losses_i.items():
                                losses[i][name].extend(value)

                else:
                    pass

                observations = [o for o in next_observations]

                # and self.start_training_time_step <= self.time_step \ #<= self.end_exploration_time_step \
                # evaluate once a month for a whole week
                if self.time_step % 168 == 0 \
                        and hasattr(self, 'start_training_time_step') \
                        and self.start_training_time_step <= self.time_step <= self.end_exploration_time_step \
                        and self.time_step > self.batch_size:
                    old_time_step = self.time_step

                    eval_env = copy.deepcopy(self.env)
                    eval_observations = eval_env.reset()

                    while not eval_env.done:
                        actions = self.predict(eval_observations, deterministic=True)
                        eval_observations, eval_rewards, _, _ = eval_env.step(actions)

                    kpis = eval_env.evaluate()
                    kpis = kpis[(kpis['cost_function'].isin(['average_daily_fossil_share',
                                                             'average_daily_fossil_share_grid',
                                                             '1 - used_pv_of_total_share',
                                                             'fossil_energy_consumption']))].dropna()
                    kpis['value'] = kpis['value'].round(3)
                    kpis = kpis.rename(columns={'cost_function': 'kpi'})
                    kpis = kpis[kpis['level'] == 'district'].copy()

                    for kpi, value in zip(kpis['kpi'], kpis['value']):
                        if not isinstance(value, float):
                            value = value[0]

                        eval_results[kpi].append(value)

                        if kpi == 'fossil_energy_consumption' and value < kpi_min:
                            best_state = copy.deepcopy(self)
                            kpi_min = value

                    self.time_step = old_time_step

                logging.debug(
                    f'Time step: {self.env.time_step}/{self.env.time_steps - 1},' \
                    f' Episode: {episode}/{episodes - 1},' \
                    f' Actions: {actions},' \
                    f' Rewards: {new_rewards}'
                )

            # store episode's env to disk
            if keep_env_history:
                self.__save_env(episode, env_history_directory)
            else:
                pass

        return losses, rewards, eval_results, best_state

    def get_env_history(self, directory: Path, episodes: List[int] = None):
        env_history = ()
        episodes = sorted([
            int(f.split(directory)[-1].split('.')[0]) for f in os.listdir(directory) if f.endswith('.pkl')
        ]) if episodes is None else episodes

        for episode in episodes:
            filepath = os.path.join(directory, f'{int(episode)}.pkl')

            with (open(filepath, 'rb')) as f:
                env_history += (pickle.load(f),)

        return env_history

    def __save_env(self, episode: int, directory: Path):
        filepath = os.path.join(directory, f'{int(episode)}.pkl')

        with open(filepath, 'wb') as f:
            pickle.dump(self.env, f)

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        """Provide actions for current time step.

        Return randomly sampled actions from `action_space`.

        Parameters
        ----------
        observations: List[List[float]]
            Environment observations
        deterministic: bool, default: False
            Wether to return purely exploitatative deterministic actions.

        Returns
        -------
        actions: List[float]
            Action values
        """

        actions = [list(s.sample()) for s in self.action_space]
        self.actions = actions
        self.next_time_step()
        return actions

    def __set_logger(self, logging_level: int = None):
        logging_level = 30 if logging_level is None else logging_level
        assert logging_level >= 0, 'logging_level must be >= 0'
        LOGGER.setLevel(logging_level)

    def update(self, *args, **kwargs) -> Mapping[str, List[float]]:
        """Update replay buffer and networks.

        Return value
        ------
        losses: Mapping[str, List[float]]
            Mapping of neural-network name to loss values of training steps.

        Notes
        -----
        This implementation does nothing but is kept to keep the API for all agents similar during simulation.
        """

        pass

    def next_time_step(self):
        super().next_time_step()

        for i in range(len(self.action_space)):
            self.__actions[i].append([])

    def reset(self):
        super().reset()
        self.__actions = [[[]] for _ in self.action_space]
