import random
from typing import Tuple

import numpy as np

# conditional imports
try:
    import torch
    from torch.distributions import Normal
    from torch.nn import init
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise Exception("This functionality requires you to install torch. You can install torch by : pip install torch torchvision, or for more detailed instructions please visit https://pytorch.org.")

class PolicyNetwork(nn.Module):
    def __init__(self, 
                 num_inputs, 
                 num_actions, 
                 action_space, 
                 action_scaling_coef, 
                 hidden_dim = [400,300],
                 kaiming_initialization = False,
                 init_w = 3e-3, 
                 log_std_min = -20, 
                 log_std_max = 2, 
                 epsilon = 1e-6):
        
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.epsilon = epsilon
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])

        self.mean_linear = nn.Linear(hidden_dim[1], num_actions)
        self.log_std_linear = nn.Linear(hidden_dim[1], num_actions)

        if kaiming_initialization:
            init.kaiming_normal_(self.mean_linear.weight, mode='fan_in')
            init.kaiming_normal_(self.log_std_linear.weight, mode='fan_in')
        else:
            self.mean_linear.weight.data.uniform_(-init_w, init_w)
            self.mean_linear.bias.data.uniform_(-init_w, init_w)

            self.log_std_linear.weight.data.uniform_(-init_w, init_w)
            self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_scale = torch.FloatTensor(
            action_scaling_coef * (action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor(
            action_scaling_coef * (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        if deterministic:
            action = mean

        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(PolicyNetwork, self).to(device)
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, alpha=0.6, beta=0.4, beta_annealing=0.0001, **kwargs):
        """ Initialize Prioritized Replay Buffer as described in
            Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).

        :param alpha: float
            Prioritization of transitions degree
        :param beta: float
            Initial importance sampling correction degree
        :param beta_annealing: float
            Factor to anneal beta over time
        """
        super(PrioritizedReplayBuffer, self).__init__(**kwargs)

        self.priorities = []
        self.alpha = alpha
        self.beta_0 = beta
        self.beta_annealing = beta_annealing
        self.transitions = None

    def push(self, state, action, reward, next_state, done):
        if self.position == 0:
            max_prio = 1e-5  # not 0 to prevent numerical errors
        else:
            max_prio = max(self.priorities)

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.priorities.append(None)

        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=1):
        if batch_size > self.capacity:
            batch_size = self.capacity

        probabilities = np.array(self.priorities[:self.capacity]) ** self.alpha
        P = probabilities / probabilities.sum()

        inds = np.random.choice(range(len(self.buffer)), batch_size, p=P)

        # beta annealing
        beta = min(1, self.beta_0 + (1. - self.beta_0) * self.beta_annealing)

        weights = (len(self.buffer) * P[inds]) ** (-beta)
        weights = np.array(weights / weights.max())

        return [self.buffer[i] for i in inds], inds, weights

    def update_priorities(self, indices, new_priorities):
        """

        :param indices: np.array
            indices of the transitions whose priorities should be updated
        :param new_priorities: np.array
        :return:
        """
        self.priorities[indices] = new_priorities


class RegressionBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.x = []
        self.y = []
        self.position = 0
    
    def push(self, variables, targets):
        if len(self.x) < self.capacity and len(self.x)==len(self.y):
            self.x.append(None)
            self.y.append(None)
        
        self.x[self.position] = variables
        self.y[self.position] = targets
        self.position = (self.position + 1) % self.capacity
    
    def __len__(self):
        return len(self.x)
    
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=[400,300], kaiming_initialization=False, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], 1)
        self.ln1 = nn.LayerNorm(hidden_size[0])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        if kaiming_initialization:
            init.kaiming_normal_(self.linear3.weight, mode='fan_in')
            init.kaiming_normal_(self.linear3.weight, mode='fan_in')
        else:
            self.linear3.weight.data.uniform_(-init_w, init_w)
            self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.ln1(F.relu(self.linear1(x)))
        x = self.ln2(F.relu(self.linear2(x)))
        x = self.linear3(x)
        return x