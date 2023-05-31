from typing import List
import numpy as np
import numpy.typing as npt

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except (ModuleNotFoundError, ImportError) as e:
    raise Exception(
        "This functionality requires you to install torch. You can install torch by : pip install torch torchvision, or for more detailed instructions please visit https://pytorch.org.")

from citylearn.agents.sac import SAC

class SACDB2(SAC):
    def __init__(self, *args, imitation_lr: float = 0.01, **kwargs):
        r"""Initialize :class:`SACDB2`.

        Parameters
        ----------
        *args : tuple
            `SAC` positional arguments.
        imitation_lr: float
            Imitation learning rate

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """
        super().__init__(*args, **kwargs)

        self.imitation_lr = imitation_lr
        self.demonstrator_policy_net = [None for _ in range(self.env.demonstrator_count)]

        self.set_demonstrator_policies()

    def update(self, observations: List[List[float]], actions: List[List[float]], reward: List[float],
               next_observations: List[List[float]], done: bool):
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
        """

        # Run once the regression model has been fitted
        # Normalize all the observations using periodical normalization, one-hot encoding, or -1, 1 scaling. It also removes observations that are not necessary (solar irradiance if there are no solar PV panels).

        for i, (o, a, r, n) in enumerate(zip(observations, actions, reward, next_observations)):
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
                    o = self.update_step(i)

                    # Use demonstrator actions for updating policy, only if the building is not a demonstrator itself
                    if not self.env.buildings[i].demonstrator:
                        for demonstrator_policy in self.demonstrator_policy_net:
                            demonstrator_actions, log_pi, _ = demonstrator_policy.sample(o)
                            q_demonstrator = torch.min(
                                self.soft_q_net1[i](o, demonstrator_actions),
                                self.soft_q_net2[i](o, demonstrator_actions)
                            )
                            q_demonstrator = q_demonstrator + self.imitation_lr * (1-q_demonstrator)
                            policy_loss = (self.alpha * log_pi - q_demonstrator).mean()
                            self.policy_optimizer[i].zero_grad()
                            policy_loss.backward()
                            self.policy_optimizer[i].step()

            else:
                pass

    def set_demonstrator_policies(self):
        demonstrator_count = 0
        for i in range(len(self.action_dimension)):
            if self.env.buildings[i].demonstrator:
                self.demonstrator_policy_net[demonstrator_count] = self.policy_net[i]
                demonstrator_count += 1


