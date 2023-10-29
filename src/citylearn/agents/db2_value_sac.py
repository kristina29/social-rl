from typing import List, Mapping

import torch
from torch import tensor, nn

from citylearn.agents.sac import SAC


class SACDB2VALUE(SAC):
    def __init__(self, *args, imitation_lr: float = 0.01, pretrained_demonstrator: SAC = None,
                 deterministic_demo: bool = False, extra_policy_update: bool = False, **kwargs):
        r"""Initialize :class:`SACDB2`.

        Parameters
        ----------
        *args : tuple
            `SAC` positional arguments.
        imitation_lr: float
            Imitation learning rate
        pretrained_demonstrator: SAC
            Pretrained SAC agent to use as demonstrator
        deterministic_demo: bool
            Use deterministic or sampled actions of the demonstrator
        extra_policy_update: bool
            Update the policy after the social Q-Value update

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """
        super().__init__(*args, **kwargs)

        self.imitation_lr = imitation_lr
        self.pretrained_demonstrator = pretrained_demonstrator
        self.deterministic_demo = deterministic_demo
        self.extra_policy_update = extra_policy_update

        self.demonstrator_policy_net = [None for _ in range(self.env.demonstrator_count)]

        self.set_demonstrator_policies()

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
        ------
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
                    o, q1_loss, q2_loss, policy_loss, alpha_loss = self.update_step(i)
                    current_losses['q1_losses'].append(q1_loss)
                    current_losses['q2_losses'].append(q2_loss)
                    current_losses['policy_losses'].append(policy_loss)
                    current_losses['alpha_losses'].append(alpha_loss)

                    # Use demonstrator actions for updating Q-Value network
                    for demonstrator_policy in self.demonstrator_policy_net:
                        self.social_q_value_update(i, o, demonstrator_policy)
            else:
                pass

            losses[i] = current_losses

        return losses

    def social_q_value_update(self, i, state, demonstrator_policy):
        with torch.no_grad():
            demo_state = state
            if self.n_interchanged_obs > 0 and self.pretrained_demonstrator is not None:
                demo_state = state[:, :-self.n_interchanged_obs]
            demonstrator_actions, log_pi, _ = demonstrator_policy.sample(demo_state, self.deterministic_demo)

            target_q_values = torch.min(
                self.target_soft_q_net1[i](state, demonstrator_actions),
                self.target_soft_q_net2[i](state, demonstrator_actions),
            )
            q_target = target_q_values + self.imitation_lr * torch.abs(target_q_values)

        # Update Soft Q-Networks
        q1_pred = self.soft_q_net1[i](state, demonstrator_actions)
        q2_pred = self.soft_q_net2[i](state, demonstrator_actions)

        q1_loss = self.soft_q_criterion(q1_pred, q_target)
        q2_loss = self.soft_q_criterion(q2_pred, q_target)

        self.soft_q_optimizer1[i].zero_grad()
        q1_loss.backward()
        self.soft_q_optimizer1[i].step()

        self.soft_q_optimizer2[i].zero_grad()
        q2_loss.backward()
        self.soft_q_optimizer2[i].step()

        # Soft Updates
        self.update_targets(i)

        if self.extra_policy_update:
            # Update Policy
            self.policy_update(i, state)

    def set_demonstrator_policies(self):
        demonstrator_count = 0
        if self.pretrained_demonstrator is not None:
            self.demonstrator_policy_net[demonstrator_count] = self.pretrained_demonstrator.policy_net[0]
        else:
            for i in range(len(self.action_dimension)):
                if self.env.buildings[i].demonstrator:
                    self.demonstrator_policy_net[demonstrator_count] = self.policy_net[i]
                    demonstrator_count += 1
