import pickle
from typing import List

import numpy as np
from matplotlib import pyplot as plt, rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def plot_losses(losses, include_alpha, mode) -> List[plt.Figure]:
    if mode == 1:
        fig, ax = plt.subplots(figsize=(14,7))
    elif mode == 2:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    else:
        fig, (ax, ax2) = plt.subplots(1,2, figsize=(14,7))
    #ax2 = ax.twinx()
    i = 0

    env_names = []

    for env_name, env_values in losses.items():
        env_names.append(env_name)
        for nn_name, nn_values in env_values.items():
            if i == 0:
                if nn_name == 'policy_losses':
                    if mode == 3:
                        i += 1
                        label = 'Policy loss'
                    else:
                        label = env_name
                    ax.plot(np.arange(len(nn_values)), nn_values, label=label, zorder=3)
                if nn_name == 'q1_losses' and mode == 3:
                    ax.plot(np.arange(len(nn_values)), nn_values, label='Q-Network loss', zorder=3)
                if include_alpha[i] and nn_name == 'alpha_vals':
                    ax2.plot(np.arange(len(nn_values)), nn_values, label=f'{env_name} (alpha vals)', linewidth=2,
                             linestyle=(5, (10, 3)))
            elif i == 1 and mode == 3:
                if nn_name == 'policy_losses':
                    ax2.plot(np.arange(len(nn_values)), nn_values, label='Policy loss', zorder=3)
                if nn_name == 'q1_losses':
                    ax2.plot(np.arange(len(nn_values)), nn_values, label='Q-Network loss', zorder=3)

    ax.axhline(0, color='black', zorder=2)
    ax.tick_params(axis='x', which='both', labelsize=21)
    ax.tick_params(axis='y', which='both', labelsize=21)
    #ax.set_title(f'{env_name}')
    ax.grid(axis='y', zorder=1)

    if mode == 2:
        ax.set_ylabel('Policy loss value', fontsize=21)
        ax2.set_ylabel(r'Temperature $\alpha$', fontsize=21, labelpad=15)
        ax2.set_xlabel('Time step', fontsize=21)
        ax2.tick_params(axis='x', which='both', labelsize=21)
        ax2.tick_params(axis='y', which='both', labelsize=21)
        ax2.grid(axis='y', zorder=1)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles=handles, bbox_to_anchor=(0.074, 0.99),
                   loc='upper left', fontsize=21)
        plt.tight_layout()
    elif mode == 3:
        ax.set_xlabel('Time step', fontsize=21)
        ax.set_title(f'a) {env_names[0]}', fontsize=21, pad=10)
        ax2.set_title(f'b) {env_names[1]}', fontsize=21, pad=10)
        ax2.axhline(0, color='black', zorder=2)
        ax2.set_xlabel('Time step', fontsize=21)
        ax2.tick_params(axis='x', which='both', labelsize=21)
        ax2.tick_params(axis='y', which='both', labelsize=21)
        ax2.grid(axis='y', zorder=1)
        ax.set_ylabel('Loss value', fontsize=21)
        ax2.set_ylabel('Loss value', fontsize=21)
        handles, labels = ax.get_legend_handles_labels()
        plt.tight_layout(rect=[0, 0.15, 1, 1])
        plt.subplots_adjust(wspace=0.3)
        fig.legend(handles=handles,  # bbox_to_anchor=(0.5, 0.1),
                   loc='lower center', fontsize=21, framealpha=0)
    else:
        ax.set_xlabel('Time step', fontsize=21)
        plt.legend(loc='upper left', fontsize=21)
        ax.set_ylabel('Policy loss value', fontsize=21)
        plt.tight_layout()

    fig.savefig('losses.pdf')
    #plt.show()


if __name__ == '__main__':
    mode = 1

    if mode == 1:
        loss_files = ['../experiments/SAC/30_renewable_prod/reward_05pvprice/0.5/losses_20231002T130931.pkl',
                      '../experiments/Imitation_Learning/demo_b5/losses_20231026T121912.pkl']
        agents = ['SAC', 'PRB_SAC']

        names = ['SAC Baseline', 'Imitation Learning using Transitions of D5']
        include_alpha = [False, False]
    elif mode == 2:
        loss_files = ['../experiments/SAC-DemoPol/32_demo_b5/ir0.2/socialMode2/losses_20231107T200708.pkl',
                      '../experiments/SAC-DemoPol/32_demo_b5/ir0.2/socialMode5/losses_20231107T201820.pkl']
        agents = ['SAC_DB2', 'SAC_DB2']

        names = ['Mode 2', 'Mode 5']
        include_alpha = [True, True]
    elif mode == 3:
        loss_files = ['../experiments/SAC_DB2Value/9_interchanged_observations/demo_b6_determ/extra_pol_update/ir0.15/losses_20231029T201710.pkl',
                      '../experiments/SAC_DB2Value/4_demo_b6_policyupdate/ir0.0001/losses_20231003T110826.pkl' ]
        agents = ['SAC_DB2Value', 'SAC_DB2Value']

        names = [r'$\alpha_i=0.15$', r'$\alpha_i=1e^{-4}$']
        include_alpha = [False, False]

    final_losses = {}

    for i, file in enumerate(loss_files):
        with open(file, 'rb') as file:
            losses = pickle.load(file)

        final_losses[names[i]] = losses[agents[i]][0]

    plot_losses(final_losses, include_alpha, mode)