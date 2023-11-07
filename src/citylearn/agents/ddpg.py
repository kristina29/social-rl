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

from citylearn.agents.rlc import RLC
from citylearn.preprocessing import Encoder, RemoveFeature
from citylearn.rl import ReplayBuffer, SoftQNetwork, DDPGActor, PrioritizedReplayBuffer


class DDPG(RLC):
    def __init__(self, l2_loss: bool=False, target_policy_noise: float=0.2, target_noise_clip: float=0.5, *args, **kwargs):
        r"""Initialize :class:`DDPG`.

        Parameters
        ----------
        *args : tuple
            `RLC` positional arguments.
        target_policy_noise : Standard deviation of Gaussian noise added to target policy (smoothing noise)
        target_noise_clip : Limit for absolute value of target policy smoothing noise.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """

        super().__init__(*args, **kwargs)

        # internally defined
        self.normalized = [False for _ in self.action_space]
        self.l2_loss = l2_loss
        if l2_loss:
            self.soft_q_criterion = nn.MSELoss()
        else:
            self.soft_q_criterion = nn.SmoothL1Loss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.replay_buffer = [ReplayBuffer(int(self.replay_buffer_capacity)) for _ in self.action_space]
        self.prioritized_replay_buffer = False
        self.soft_q_net = [None for _ in self.action_space]
        self.target_soft_q_net = [None for _ in self.action_space]
        self.policy_net = [None for _ in self.action_space]
        self.target_policy_net = [None for _ in self.action_space]
        self.soft_q_optimizer = [None for _ in self.action_space]
        self.policy_optimizer = [None for _ in self.action_space]
        self.norm_mean = [None for _ in self.action_space]
        self.norm_std = [None for _ in self.action_space]
        self.r_norm_mean = [None for _ in self.action_space]
        self.r_norm_std = [None for _ in self.action_space]
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
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
            current_losses = {'q_losses': [],
                              'policy_losses': []}

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
                    _, q_loss, policy_loss = self.update_step(i)
                    current_losses['q_losses'].append(q_loss)
                    current_losses['policy_losses'].append(policy_loss)
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

    def update_step(self, i) -> (List[List[float]], float, float):
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

        # Update Q-Value network
        q_loss = self.q_value_update(i, o, a, r, d, n, inds, weights)

        # Update Policy
        policy_loss = self.policy_update(i, o)

        # Soft Updates
        self.update_targets(i)

        return o, q_loss.item(), policy_loss.item()

    def q_value_update(self, i, state, action, reward, done, next_state, inds, weights):
        target_actions = self.target_policy_net[i].sample(next_state, deterministic=False)
        target_q_values = self.target_soft_q_net[i].forward(next_state, target_actions)
        q_target = reward + (1 - done) * self.discount * target_q_values

        q_pred = self.soft_q_net[i](state, action)

        # optimize critic
        if self.prioritized_replay_buffer:
            if self.l2_loss:
                td_error = q_target - q_pred
                q_loss = 0.5 * (td_error ** 2 * weights).mean()
            else:
                q_loss, td_error = smoothl1_withweights(q_target, q_pred, weights)

            priorities = (abs(td_error) + 1e-5).squeeze().detach().numpy()
            self.replay_buffer[i].update_priorities(inds, priorities)
        else:
            q_loss = self.soft_q_criterion(q_pred, q_target)

        self.soft_q_optimizer[i].zero_grad()
        q_loss.backward()
        self.soft_q_optimizer[i].step()

        return q_loss

    def policy_update(self, i, state):
        new_actions = self.policy_net[i].forward(state)
        policy_loss = - self.soft_q_net[i].forward(state, new_actions).mean()
        self.policy_optimizer[i].zero_grad()
        policy_loss.backward()
        self.policy_optimizer[i].step()

        return policy_loss

    def update_targets(self, i):
        for target_param, param in zip(self.target_soft_q_net[i].parameters(),
                                       self.soft_q_net[i].parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        for target_param, param in zip(self.target_policy_net[i].parameters(),
                                       self.policy_net[i].parameters()):
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
            a = self.policy_net[i].sample(o, deterministic)
            actions.append(a.detach().cpu().numpy()[0])

        return actions

    def get_exploration_prediction(self, observations: List[List[float]]) -> List[List[float]]:
        """Return randomly sampled actions from `action_space`.

        Returns
        -------
        actions: List[List[float]]
            Action values.
        """

        # random actions
        return [list(s.sample()) for s in self.action_space]

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
            self.soft_q_net[i] = SoftQNetwork(observation_dimension, self.action_dimension[i],
                                              self.hidden_dimension).to(self.device)
            self.target_soft_q_net[i] = SoftQNetwork(observation_dimension, self.action_dimension[i],
                                                     self.hidden_dimension).to(self.device)

            for target_param, param in zip(self.target_soft_q_net[i].parameters(), self.soft_q_net[i].parameters()):
                target_param.data.copy_(param.data)

            # Policy
            self.policy_net[i] = DDPGActor(observation_dimension, self.action_dimension[i], self.hidden_dimension,
                                           target_noise=self.target_policy_noise,
                                           target_noise_clip=self.target_noise_clip).to(self.device)
            self.target_policy_net[i] = DDPGActor(observation_dimension, self.action_dimension[i],
                                                  self.hidden_dimension).to(self.device)
            for target_param, param in zip(self.target_policy_net[i].parameters(), self.policy_net[i].parameters()):
                target_param.data.copy_(param.data)

            self.soft_q_optimizer[i] = optim.Adam(self.soft_q_net[i].parameters(), lr=self.lr)
            self.policy_optimizer[i] = optim.Adam(self.policy_net[i].parameters(), lr=self.lr)

    def set_encoders(self) -> List[List[Encoder]]:
        encoders = super().set_encoders()

        for i, o in enumerate(self.observation_names):
            for j, n in enumerate(o):
                if n == 'net_electricity_consumption':
                    encoders[i][j] = RemoveFeature()

                else:
                    pass

        return encoders


class PRBDDPG(DDPG):
    def __init__(self, *args, demonstrator_transitions: List, **kwargs):
        r"""Initialize :class:`PRBDDPG`.

        Parameters
        ----------
        *args : tuple
            `DDPG` positional arguments.
        demonstrator_transitions: np.ndarray
            Transitions of the demonstrator: [state, action, reward, next_state, done]

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """
        super().__init__(*args, **kwargs)

        self.replay_buffer = [PrioritizedReplayBuffer(capacity=int(self.replay_buffer_capacity))
                              for _ in self.action_space]
        self.prioritized_replay_buffer = True

        self.fill_replay_buffer(demonstrator_transitions)

    def fill_replay_buffer(self, demonstrator_transitions: List):
        for buffer in self.replay_buffer:
            for transition in demonstrator_transitions:
                buffer.push(transition[0], transition[1], transition[2], transition[3], transition[4])