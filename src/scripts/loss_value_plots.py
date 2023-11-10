import pickle
from typing import List

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({
        "text.usetex": True,
        'text.latex.preamble': r"\usepackage{bm}"
    })


def plot_losses(losses, include_alpha) -> List[plt.Figure]:

    fig, (ax, ax2) = plt.subplots(1,2, figsize=(14,7))
    #ax2 = ax.twinx()
    i = 0
    for env_name, env_values in losses.items():
        for nn_name, nn_values in env_values.items():
            if i == 0:
                ax.set_ylabel(env_name, fontsize=21, fontdict=dict(weight='bold'))
                if nn_name == 'policy_losses':
                    ax.plot(np.arange(len(nn_values)), nn_values, label='Policy loss', zorder=3)
                    i += 1
                if nn_name == 'q1_losses':
                    ax.plot(np.arange(len(nn_values)), nn_values, label='Q-Network loss', zorder=3)
            elif i == 1:
                ax2.set_ylabel(env_name, fontsize=21)
                if nn_name == 'policy_losses':
                    ax2.plot(np.arange(len(nn_values)), nn_values, label='Policy loss', zorder=3)
                if nn_name == 'q1_losses':
                    ax2.plot(np.arange(len(nn_values)), nn_values, label='Q-Network loss', zorder=3)

    ax.set_title('a)', fontsize=21, pad=10)
    ax2.set_title('b)', fontsize=21, pad=10)
    ax.axhline(0, color='black', zorder=2)
    ax2.axhline(0, color='black', zorder=2)
    ax.set_xlabel('Time step', fontsize=21)
    ax2.set_xlabel('Time step', fontsize=21)
    ax.tick_params(axis='x', which='both', labelsize=21)
    ax2.tick_params(axis='x', which='both', labelsize=21)
    ax.tick_params(axis='y', which='both', labelsize=21)
    ax2.tick_params(axis='y', which='both', labelsize=21)
    #ax.set_title(f'{env_name}')
    ax.grid(axis='y', zorder=1)
    ax2.grid(axis='y', zorder=1)

    handles, labels = ax.get_legend_handles_labels()
    #handles.append(Line2D([0], [0], label=r'$\alpha$', color='black', linewidth=2, linestyle= (5, (10, 3))))

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    fig.legend(handles=handles, #bbox_to_anchor=(0.5, 0.1),
               loc='lower center', fontsize=21, framealpha=0)

    plt.subplots_adjust(wspace=0.3)


    fig.savefig('losses.pdf')
    #plt.show()


if __name__ == '__main__':
    loss_files = [#'../experiments/SAC_DB2/30_renewable_prod/reward_05pvprice/0.5/losses_20231002T130931.pkl',
                  '../experiments/SAC_DB2Value/9_interchanged_observations/demo_b6_determ/extra_pol_update/ir0.15/losses_20231029T201710.pkl',
                  '../experiments/SAC_DB2Value/4_demo_b6_policyupdate/ir0.0001/losses_20231003T110826.pkl' ]
    agents = [#'SAC',
         'SAC_DB2Value', 'SAC_DB2Value']

    names = [#'x',
         r'$\bm{\alpha_i=0.15}$', r'$\bm{\alpha_i=1e^{-4}}$']
    include_alpha = [False, False]

    final_losses = {}

    for i, file in enumerate(loss_files):
        with open(file, 'rb') as file:
            losses = pickle.load(file)

        final_losses[names[i]] = losses[agents[i]][0]

    plot_losses(final_losses, include_alpha)