from typing import List, Mapping
import numpy as np
import numpy.typing as npt

from citylearn.utilities import smoothl1_withweights

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except (ModuleNotFoundError, ImportError) as e:
    raise Exception(
        "This functionality requires you to install torch. You can install torch by : pip install torch torchvision, or for more detailed instructions please visit https://pytorch.org.")

from citylearn.agents.rbc import RBC, BasicRBC, OptimizedRBC, BasicBatteryRBC
from citylearn.agents.rlc import RLC
from citylearn.preprocessing import Encoder, RemoveFeature
from citylearn.rl import PolicyNetwork, ReplayBuffer, SoftQNetwork


class SAC(RLC):
    def __init__(self, autotune_entropy: bool=False, clip_gradient: bool=False, kaiming_initialization: bool=False,
                 l2_loss: bool=False, n_interchanged_obs: int=0, *args, **kwargs):
        r"""Initialize :class:`SAC`.

        Parameters
        ----------
        *args : tuple
            `RLC` positional arguments.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """

        super().__init__(*args, **kwargs)

        # internally defined
        self.autotune_entropy = autotune_entropy
        self.clip_gradient = clip_gradient
        self.kaiming_initialization = kaiming_initialization
        self.normalized = [False for _ in self.action_space]
        self.l2_loss = l2_loss
        self.n_interchanged_obs = n_interchanged_obs
        if l2_loss:
            self.soft_q_criterion = nn.MSELoss()
        else:
            self.soft_q_criterion = nn.SmoothL1Loss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.replay_buffer = [ReplayBuffer(int(self.replay_buffer_capacity)) for _ in self.action_space]
        self.prioritized_replay_buffer = False
        self.soft_q_net1 = [None for _ in self.action_space]
        self.soft_q_net2 = [None for _ in self.action_space]
        self.target_soft_q_net1 = [None for _ in self.action_space]
        self.target_soft_q_net2 = [None for _ in self.action_space]
        self.policy_net = [None for _ in self.action_space]
        self.soft_q_optimizer1 = [None for _ in self.action_space]
        self.soft_q_optimizer2 = [None for _ in self.action_space]
        self.policy_optimizer = [None for _ in self.action_space]
        self.target_entropy = [None for _ in self.action_space]
        self.norm_mean = [None for _ in self.action_space]
        self.norm_std = [None for _ in self.action_space]
        self.r_norm_mean = [None for _ in self.action_space]
        self.r_norm_std = [None for _ in self.action_space]
        self.alpha = [self.alpha for _ in self.action_space]
        self.log_alpha = [None for _ in self.action_space]
        self.alpha_optimizer = [None for _ in self.action_space]
        self.set_networks()

    def update(self, observations: List[List[float]], actions: List[List[float]], reward: List[float],
               next_observations: List[List[float]], done: bool) -> Mapping[int, Mapping[str, List[float]]]:
        r"""Update replay buffer.

        Parameters
        ----------
        observations : List[List[float]]
            Previous time step observations.
        actions : List[List[float]]
            Previous time step actions.
        reward : List[float]
            Current time step reward.
        next_observations : List[List[float]]
            Current time step observations.
        done : bool
            Indication that episode has ended.

        Return value
        ------------
        losses: Mapping[int, Mapping[str, List[float]]]
            Mapping of index to Mapping from neural-network name to loss values of training steps.
        """

        # Run once the regression model has been fitted
        # Normalize all the observations using periodical normalization, one-hot encoding, or -1, 1 scaling. It also removes observations that are not necessary (solar irradiance if there are no solar PV panels).
        losses = {}

        for i, (o, a, r, n) in enumerate(zip(observations, actions, reward, next_observations)):
            current_losses = {'q1_losses': [],
                              'q2_losses': [],
                              'policy_losses': [],
                              'alpha_losses': []}

            o = self.get_encoded_observations(i, o)
            n = self.get_encoded_observations(i, n)

            if self.normalized[i]:
                o = self.get_normalized_observations(i, o)
                n = self.get_normalized_observations(i, n)
                r = self.get_normalized_reward(i, r)
            else:
                pass

            self.replay_buffer[i].push(o, a, r, n, done)

            if self.time_step >= self.start_training_time_step and self.batch_size <= len(self.replay_buffer[i]):
                if not self.normalized[i]:
                    self.normalize(i)

                else:
                    pass

                for _ in range(self.update_per_time_step):
                    _, q1_loss, q2_loss, policy_loss, alpha_loss = self.update_step(i)
                    current_losses['q1_losses'].append(q1_loss)
                    current_losses['q2_losses'].append(q2_loss)
                    current_losses['policy_losses'].append(policy_loss)
                    current_losses['alpha_losses'].append(alpha_loss)
            else:
                pass

            losses[i] = current_losses

        return losses

    def normalize(self, i):
        if self.prioritized_replay_buffer:
            self.normalize_prioritized_buffer(i)

        else:
            # calculate normalized observations and rewards
            X = np.array([j[0] for j in self.replay_buffer[i].buffer], dtype=float)
            self.norm_mean[i] = np.nanmean(X, axis=0)
            self.norm_std[i] = np.nanstd(X, axis=0) + 1e-5
            R = np.array([j[2] for j in self.replay_buffer[i].buffer], dtype=float)
            self.r_norm_mean[i] = np.nanmean(R, dtype=float)
            self.r_norm_std[i] = np.nanstd(R, dtype=float) / self.reward_scaling + 1e-5

            # update buffer with normalization
            self.replay_buffer[i].buffer = [(
                np.hstack(self.get_normalized_observations(i, o).reshape(1, -1)[0]),
                a,
                self.get_normalized_reward(i, r),
                np.hstack(self.get_normalized_observations(i, n).reshape(1, -1)[0]),
                d
            ) for o, a, r, n, d in self.replay_buffer[i].buffer]

        self.normalized[i] = True

    def normalize_prioritized_buffer(self, i):
        # calculate normalized observations and rewards
        X = np.array([j[0] for j in self.replay_buffer[i].buffer], dtype=float)
        self.norm_mean[i] = np.nanmean(X, axis=0)
        self.norm_std[i] = np.nanstd(X, axis=0) + 1e-5
        R = np.array([j[2] for j in self.replay_buffer[i].buffer], dtype=float)
        self.r_norm_mean[i] = np.nanmean(R, dtype=float)
        self.r_norm_std[i] = np.nanstd(R, dtype=float) / self.reward_scaling + 1e-5

        # update buffer with normalization
        for j, transition in enumerate(self.replay_buffer[i].buffer):
            o = transition[0]
            a = transition[1]
            r = transition[2]
            n = transition[3]
            d = transition[4]

            self.replay_buffer[i].update_transition(j,
                                                    np.hstack(self.get_normalized_observations(i, o).reshape(1, -1)[0]),
                                                    a,
                                                    self.get_normalized_reward(i, r),
                                                    np.hstack(self.get_normalized_observations(i, n).reshape(1, -1)[0]),
                                                    d)

    def update_step(self, i) -> (List[List[float]], float, float, float, float):
        if self.prioritized_replay_buffer:
            transitions, inds, weights = self.replay_buffer[i].sample(self.batch_size)
            weights = torch.FloatTensor(weights).unsqueeze(1)
            o = torch.FloatTensor(np.stack(transitions[:, 0]))
            a = torch.FloatTensor(np.stack(transitions[:, 1])[:, None]).squeeze(dim=1)
            r = torch.FloatTensor(np.stack(transitions[:, 2]))
            n = torch.FloatTensor(np.stack(transitions[:, 3]))
            d = torch.FloatTensor(np.stack(transitions[:, 4]))
        else:
            o, a, r, n, d = self.replay_buffer[i].sample(self.batch_size)
            inds = None
            weights = None

        tensor = torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor
        o = tensor(o).to(self.device)
        n = tensor(n).to(self.device)
        a = tensor(a).to(self.device)
        r = tensor(r).unsqueeze(1).to(self.device)
        d = tensor(d).unsqueeze(1).to(self.device)

        # Update Q-Value networks
        q1_loss, q2_loss = self.q_value_update(i, o, a, r, d, n, inds, weights)

        # Update Policy
        log_pi, policy_loss = self.policy_update(i, o)

        # Update Entropy
        alpha_loss = self.alpha_update(i, log_pi)

        # Soft Updates
        self.update_targets(i)

        return o, q1_loss.item(), q2_loss.item(), policy_loss.item(), alpha_loss.item()

    def q_value_update(self, i, state, action, reward, done, next_state, inds, weights):
        with torch.no_grad():
            # Update Q-values. First, sample an action from the Gaussian policy/distribution for the current (next)
            # observation and its associated log probability of occurrence.
            new_next_actions, new_log_pi, _ = self.policy_net[i].sample(next_state)

            # The updated Q-value is found by subtracting the logprob of the sampled action (proportional to the
            # entropy) to the Q-values estimated by the target networks.
            target_q_values = torch.min(
                self.target_soft_q_net1[i](next_state, new_next_actions),
                self.target_soft_q_net2[i](next_state, new_next_actions),
            ) - self.alpha[i] * new_log_pi
            q_target = reward + (1 - done) * self.discount * target_q_values

        # Update Soft Q-Networks
        q1_pred = self.soft_q_net1[i](state, action)
        q2_pred = self.soft_q_net2[i](state, action)

        if self.prioritized_replay_buffer:
            if self.l2_loss:
                td_error1 = q_target - q1_pred
                td_error2 = q_target - q2_pred
                q1_loss = 0.5 * (td_error1 ** 2 * weights).mean()
                q2_loss = 0.5 * (td_error2 ** 2 * weights).mean()
            else:
                q1_loss, td_error1 = smoothl1_withweights(q_target, q1_pred, weights)
                q2_loss, td_error2 = smoothl1_withweights(q_target, q2_pred, weights)

            priorities = ((abs(td_error1) + abs(td_error2)) / 2 + 1e-5).squeeze().detach().numpy()
            self.replay_buffer[i].update_priorities(inds, priorities)
        else:
            q1_loss = self.soft_q_criterion(q1_pred, q_target)
            q2_loss = self.soft_q_criterion(q2_pred, q_target)

        self.soft_q_optimizer1[i].zero_grad()
        q1_loss.backward()

        # Gradient Value Clipping
        if self.clip_gradient:
            nn.utils.clip_grad_value_(self.soft_q_net1[i].parameters(), clip_value=1.0)

        self.soft_q_optimizer1[i].step()

        self.soft_q_optimizer2[i].zero_grad()
        q2_loss.backward()

        # Gradient Value Clipping
        if self.clip_gradient:
            nn.utils.clip_grad_value_(self.soft_q_net2[i].parameters(), clip_value=1.0)
        self.soft_q_optimizer2[i].step()

        return q1_loss, q2_loss

    def policy_update(self, i, state):
        new_actions, log_pi, _ = self.policy_net[i].sample(state)
        q_new_actions = torch.min(
            self.soft_q_net1[i](state, new_actions),
            self.soft_q_net2[i](state, new_actions)
        )
        policy_loss = (self.alpha[i] * log_pi - q_new_actions).mean()
        self.policy_optimizer[i].zero_grad()
        policy_loss.backward()

        # Gradient Value Clipping
        if self.clip_gradient:
            nn.utils.clip_grad_value_(self.policy_net[i].parameters(), clip_value=1.0)

        self.policy_optimizer[i].step()

        return log_pi, policy_loss

    def alpha_update(self, i, log_pi):
        if self.autotune_entropy:
            alpha_loss = (-self.log_alpha[i] * (log_pi + self.target_entropy[i]).detach()).mean()

            self.alpha_optimizer[i].zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer[i].step()
            self.alpha[i] = self.log_alpha[i].exp().item()
        else:
            alpha_loss = torch.tensor(0.)

        return alpha_loss

    def update_targets(self, i):
        for target_param, param in zip(self.target_soft_q_net1[i].parameters(),
                                       self.soft_q_net1[i].parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        for target_param, param in zip(self.target_soft_q_net2[i].parameters(),
                                       self.soft_q_net2[i].parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def predict(self, observations: List[List[float]], deterministic: bool = None):
        r"""Provide actions for current time step.

        Will return randomly sampled actions from `action_space` if :attr:`end_exploration_time_step` >= :attr:`time_step`
        else will use policy to sample actions.

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

        deterministic = False if deterministic is None else deterministic

        if self.time_step > self.end_exploration_time_step or deterministic:
            actions = self.get_post_exploration_prediction(observations, deterministic)

        else:
            actions = self.get_exploration_prediction(observations)

        actions = np.array(actions)
        actions = list(np.nan_to_num(actions, 0.))
        for i, a in enumerate(actions):
            actions[i] = list(a)
        self.actions = actions
        self.next_time_step()
        return actions

    def get_post_exploration_prediction(self, observations: List[List[float]], deterministic: bool) -> List[
        List[float]]:
        """Action sampling using policy, post-exploration time step"""

        actions = []

        for i, o in enumerate(observations):
            o = self.get_encoded_observations(i, o)
            o = self.get_normalized_observations(i, o)
            o = torch.FloatTensor(o).unsqueeze(0).to(self.device)
            result = self.policy_net[i].sample(o)
            a = result[2] if self.time_step >= self.deterministic_start_time_step or deterministic else result[0]
            actions.append(a.detach().cpu().numpy()[0])

        return actions

    def get_exploration_prediction(self, observations: List[List[float]]) -> List[List[float]]:
        """Return randomly sampled actions from `action_space` multiplied by :attr:`action_scaling_coefficient`.

        Returns
        -------
        actions: List[List[float]]
            Action values.
        """

        # random actions
        return [list(self.action_scaling_coefficient * s.sample()) for s in self.action_space]

    def get_normalized_reward(self, index: int, reward: float) -> float:
        return (reward - self.r_norm_mean[index]) / self.r_norm_std[index]

    def get_normalized_observations(self, index: int, observations: List[float]) -> npt.NDArray[np.float64]:
        return (np.array(observations, dtype=float) - self.norm_mean[index]) / self.norm_std[index]

    def get_encoded_observations(self, index: int, observations: List[float]) -> npt.NDArray[np.float64]:
        return np.array([j for j in np.hstack(self.encoders[index] * np.array(observations, dtype=float)) if j != None],
                        dtype=float)

    def set_networks(self, internal_observation_count: int = None):
        internal_observation_count = 0 if internal_observation_count is None else internal_observation_count

        for i in range(len(self.action_dimension)):
            observation_dimension = self.observation_dimension[i] + internal_observation_count
            # init networks
            self.soft_q_net1[i] = SoftQNetwork(observation_dimension, self.action_dimension[i],
                                               self.hidden_dimension, self.kaiming_initialization).to(self.device)
            self.soft_q_net2[i] = SoftQNetwork(observation_dimension, self.action_dimension[i],
                                               self.hidden_dimension).to(self.device)
            self.target_soft_q_net1[i] = SoftQNetwork(observation_dimension, self.action_dimension[i],
                                                      self.hidden_dimension).to(self.device)
            self.target_soft_q_net2[i] = SoftQNetwork(observation_dimension, self.action_dimension[i],
                                                      self.hidden_dimension).to(self.device)

            for target_param, param in zip(self.target_soft_q_net1[i].parameters(), self.soft_q_net1[i].parameters()):
                target_param.data.copy_(param.data)

            for target_param, param in zip(self.target_soft_q_net2[i].parameters(), self.soft_q_net2[i].parameters()):
                target_param.data.copy_(param.data)

            # Policy
            self.policy_net[i] = PolicyNetwork(observation_dimension, self.action_dimension[i], self.action_space[i],
                                               self.action_scaling_coefficient, self.hidden_dimension).to(self.device)
            self.soft_q_optimizer1[i] = optim.Adam(self.soft_q_net1[i].parameters(), lr=self.lr)
            self.soft_q_optimizer2[i] = optim.Adam(self.soft_q_net2[i].parameters(), lr=self.lr)
            self.policy_optimizer[i] = optim.Adam(self.policy_net[i].parameters(), lr=self.lr)

            # Based on https://docs.cleanrl.dev/rl-algorithms/sac/#implementation-details
            if self.autotune_entropy:
                self.target_entropy[i] = -torch.Tensor(self.action_space[i].shape)
                self.log_alpha[i] = torch.zeros(1, requires_grad=True)
                self.alpha[i] = self.log_alpha[i].exp().item()
                self.alpha_optimizer[i] = optim.Adam([self.log_alpha[i]], lr=self.lr)


    def set_encoders(self) -> List[List[Encoder]]:
        encoders = super().set_encoders()

        for i, o in enumerate(self.observation_names):
            for j, n in enumerate(o):
                if n == 'net_electricity_consumption':
                    encoders[i][j] = RemoveFeature()

                else:
                    pass

        return encoders


class SACRBC(SAC):
    def __init__(self, *args, **kwargs):
        r"""Initialize `SACRBC`.

        Uses :class:`RBC` to select action during exploration before using :class:`SAC`.

        Parameters
        ----------
        *args : tuple
            :class:`SAC` positional arguments.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """

        super().__init__(*args, **kwargs)
        self.rbc = RBC(*args, **kwargs)

    @property
    def rbc(self) -> RBC:
        """:class:`RBC` or child class, used to select actions during exploration."""

        return self.__rbc

    @rbc.setter
    def rbc(self, rbc: RBC):
        self.__rbc = rbc

    def get_exploration_prediction(self, states: List[float]) -> List[float]:
        """Return actions using :class:`RBC`.

        Returns
        -------
        actions: List[float]
            Action values.
        """

        return self.rbc.predict(states)


class SACBasicRBC(SACRBC):
    def __init__(self, *args, **kwargs):
        r"""Initialize `SACRBC`.

        Uses :class:`BasicRBC` to select action during exploration before using :class:`SAC`.

        Parameters
        ----------
        *args : tuple
            :class:`SAC` positional arguments.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """

        super().__init__(*args, **kwargs)
        self.rbc = BasicRBC(*args, **kwargs)


class SACOptimizedRBC(SACBasicRBC):
    def __init__(self, *args, **kwargs):
        r"""Initialize `SACOptimizedRBC`.

        Uses :class:`OptimizedRBC` to select action during exploration before using :class:`SAC`.

        Parameters
        ----------
        *args : tuple
            :class:`SAC` positional arguments.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """

        super().__init__(*args, **kwargs)
        self.rbc = OptimizedRBC(*args, **kwargs)


class SACBasicBatteryRBC(SACBasicRBC):
    def __init__(self, *args, **kwargs):
        r"""Initialize `SACOptimizedRBC`.

        Uses :class:`OptimizedRBC` to select action during exploration before using :class:`SAC`.

        Parameters
        ----------
        *args : tuple
            :class:`SAC` positional arguments.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """

        super().__init__(*args, **kwargs)
        self.rbc = BasicBatteryRBC(*args, **kwargs)