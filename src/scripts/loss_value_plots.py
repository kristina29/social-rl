import pickle
from collections import Mapping
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })

def plot_losses(losses, include_alpha) -> List[plt.Figure]:

    fig, (ax, ax2) = plt.subplots(2,1, figsize=(14,12), sharex=True)
    #ax2 = ax.twinx()
    i = 0
    for env_name, env_values in losses.items():
        for nn_name, nn_values in env_values.items():
            if nn_name == 'policy_losses':
                ax.plot(np.arange(len(nn_values)), nn_values, label=env_name)
            if include_alpha[i] and nn_name == 'alpha_vals':
                ax2.plot(np.arange(len(nn_values)), nn_values, label=f'{env_name} (alpha vals)', linewidth=2, linestyle= (5, (10, 3)))
        i += 1

    ax.set_ylabel(f'Policy loss value', fontsize=21)
    ax2.set_ylabel(r'Entropy $\alpha$', fontsize=21)
    ax2.set_xlabel('Time step', fontsize=21)
    ax2.tick_params(axis='x', which='both', labelsize=21)
    ax.tick_params(axis='y', which='both', labelsize=21)
    ax2.tick_params(axis='y', which='both', labelsize=21)
    #ax.set_title(f'{env_name}')
    ax.grid(axis='y')
    ax2.grid(axis='y')

    handles, labels = ax.get_legend_handles_labels()
    #handles.append(Line2D([0], [0], label=r'$\alpha$', color='black', linewidth=2, linestyle= (5, (10, 3))))

    fig.legend(handles=handles, fontsize=21)
    plt.tight_layout()

    fig.savefig('losses.pdf')
    #plt.show()


if __name__ == '__main__':
    loss_files = [#'../experiments/SAC_DB2/30_renewable_prod/reward_05pvprice/0.5/losses_20231002T130931.pkl',
                  '../experiments/SAC_DB2/32_demo_b5/ir0.2/socialMode2/losses_20231107T200708.pkl',
                  '../experiments/SAC_DB2/32_demo_b5/ir0.2/socialMode5/losses_20231107T201820.pkl']
    agents = [#'SAC',
         'SAC_DB2', 'SAC_DB2']
    names = [#'x',
         'Mode 2', 'Mode 5']
    include_alpha = [True, True]

    final_losses = {}

    for i, file in enumerate(loss_files):
        with open(file, 'rb') as file:
            losses = pickle.load(file)

        final_losses[names[i]] = losses[agents[i]][0]

    plot_losses(final_losses, include_alpha)