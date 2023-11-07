import pickle
from collections import Mapping
from typing import List

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })

def plot_losses(losses: Mapping[str, Mapping[int, Mapping[str, List[float]]]]) -> List[plt.Figure]:

    fig, ax = plt.subplots(figsize=(14,8))
    i = 0
    for env_name, env_values in losses.items():
        for nn_name, nn_values in env_values.items():
            if nn_name == 'policy_losses':
                ax.plot(np.arange(len(nn_values)), nn_values, label=env_name)
        i += 1

    ax.set_ylabel(f'Policy loss value', fontsize=21)
    ax.set_xlabel('Time step', fontsize=21)
    ax.tick_params(axis='x', which='both', labelsize=21)
    ax.tick_params(axis='y', which='both', labelsize=21)
    #ax.set_title(f'{env_name}')
    ax.grid(axis='y')
    ax.legend(fontsize=21)

    fig.savefig('losses.pdf')
    #plt.show()


if __name__ == '__main__':
    loss_files = ['../experiments/SAC_DB2/30_renewable_prod/reward_05pvprice/0.5/losses_20231002T130931.pkl',
                  '../experiments/SAC_DB2/32_demo_b5/ir0.2/socialMode2/losses_20231002T160209.pkl',
                  '../experiments/SAC_DB2/32_demo_b5/ir0.2/socialMode5/losses_20231002T160836.pkl']
    agents = ['SAC', 'SAC_DB2', 'SAC_DB2']
    names = ['x', 'Mode 2', 'Mode 5']

    final_losses = {}

    for i, file in enumerate(loss_files):
        with open(file, 'rb') as file:
            losses = pickle.load(file)

        final_losses[names[i]] = losses[agents[i]][0]

    plot_losses(final_losses)