from typing import List, Mapping

import numpy as np
import pandas as pd

from citylearn.rl import PrioritizedReplayBuffer

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except (ModuleNotFoundError, ImportError) as e:
    raise Exception(
        "This functionality requires you to install torch. You can install torch by : pip install torch torchvision, or for more detailed instructions please visit https://pytorch.org.")

from citylearn.agents.sac import SAC


class PRBSAC(SAC):
    def __init__(self, *args, demonstrator_transitions: np.ndarray, **kwargs):
        r"""Initialize :class:`SACDB2`.

        Parameters
        ----------
        *args : tuple
            `SAC` positional arguments.
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
