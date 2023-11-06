import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

BEST_SAC_VALUE = 0.9298
BEST_SAC_B3 = 0.94
BEST_SAC_B5 = 0.957
Y_LIM = [0.91, 1.005]

sacdb2value_dirs = pd.DataFrame({'paths': ['9_interchanged_observations/random_d2/no_extra_pol',
                                           '9_interchanged_observations/random_d2/extra_pol',
                                           '9_interchanged_observations/random_d2_determ/no_extra_pol',
                                           '9_interchanged_observations/random_d2_determ/extra_pol',
                                           '1_randomdemo/d2',
                                           '1_randomdemo/d2_extrapol',
                                           '1_randomdemo/d4',
                                           '1_randomdemo/d4_extrapol',
                                           '2_demo_b6',
                                           '4_demo_b6_policyupdate',
                                           '8_determ_actions/demo_b6/non_extra_pol_update',
                                           '8_determ_actions/demo_b6/extra_pol_update',
                                           '3_demo_b5',
                                           '5_demo_b5_policyupdate',
                                           '9_interchanged_observations/demo_b6/no_extra_policy_update',
                                           '9_interchanged_observations/demo_b6/extra_policy_update',
                                           '9_interchanged_observations/demo_b6_determ/no_extra_pol_update',
                                           '9_interchanged_observations/demo_b6_determ/extra_pol_update',
                                           '7_shifted_demos/b5_demo_b5/non_extra_pol',
                                           '7_shifted_demos/b5_demo_b5/extra_pol',
                                           '7_shifted_demos/b5_demo_b6/non_extra_pol',
                                           '7_shifted_demos/b5_demo_b6/extra_pol',
                                           '7_shifted_demos/b3_demo_b3/non_extra_pol',
                                           '7_shifted_demos/b3_demo_b3/extra_pol',
                                           '7_shifted_demos/b3_demo_b6/non_extra_pol',
                                           '7_shifted_demos/b3_demo_b6/extra_pol',
                                           ],
                                 'demos': ['2 random\n(shared obs.)', '2 random\n(shared obs.)',
                                           '2 random\n(shared obs.,\ndeterm.)', '2 random\n(shared obs.,\ndeterm.)',
                                           '2 random', '2 random',
                                           '4 random', '4 random',
                                           'B6', 'B6',
                                           'B6 (determ.)', 'B6 (determ.)',
                                           'B5', 'B5',
                                           'B6\n(shared obs.)', 'B6\n(shared obs.)',
                                           'B6\n(shared obs.,\ndeterm.)', 'B6\n(shared obs.,\ndeterm.)',
                                           'B5 (only B5s)', 'B5 (only B5s)',
                                           'B6 (only B5s)', 'B6 (only B5s)',
                                           'B3 (only B3s)', 'B3 (only B3s)',
                                           'B6 (only B3s)', 'B6 (only B3s)'
                                           ],
                                 'extra_pols': [0, 1, 0, 1, #interchanged obs
                                                0, 1, 0, 1, #random demo
                                                0, 1, 0, 1, #b6
                                                0, 1,       #b5
                                                0, 1, 0, 1, #interchnaged obs
                                                0, 1, 0, 1, 0, 1, 0, 1 #shifted
                                                ],
                                 'shared_obs': [1, 1, 1, 1,
                                                0, 0, 0, 0,
                                                0, 0, 0, 0,
                                                0, 0,
                                                1, 1, 1, 1,
                                                0, 0, 0, 0, 0, 0, 0, 0],
                                 'shifted': [0, 0, 0, 0,
                                             0, 0, 0, 0,
                                             0, 0, 0, 0,
                                             0, 0,
                                             0, 0, 0, 0,
                                             1, 1, 1, 1, 1, 1, 1, 1]
                                 })



def generate_sacdb2value():
    dirs = sacdb2value_dirs

    fig, ax = plt.subplots(figsize=(15,8))
    fig2, ax2 = plt.subplots(figsize=(15,8))
    fig3, ax3 = plt.subplots(figsize=(15,8))

    i = 0

    ax1_irs = []
    ax2_irs = []
    ax3_irs = []

    for dir in dirs.iterrows():
        demo = dir[1]['demos']
        extra_pol = dir[1]['extra_pols']

        if i==0:
            temp_irs = []
            temp_foss = []
            temp_foss1 = []

        for ir in [0.0001, 0.001, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.8]:
            # for ir in [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.4]:#, 0.6, 0.8]:
            try:
                file = glob.glob(f'../experiments/SAC_DB2Value/{dir[1]["paths"]}/ir{ir}/kpis_*.csv')[0]
                kpis = pd.read_csv(file)
                kpis = kpis.set_index('kpi')
                kpis = kpis[(kpis['env_id'] == 'SAC_DB2Value Best') & (kpis['level'] == 'district')]
                v = round(kpis.loc['fossil_energy_consumption', 'net_value'] /
                          kpis.loc['fossil_energy_consumption', 'net_value_without_storage'], 3)

                if i == 0:
                    temp_foss.append(v)
                    temp_irs.append(ir)
                if i == 1:
                    temp_foss1.append(v)
            except:
                print('Missing')

        if i == 1 and dir[1]['shared_obs'] == 1:
            means = np.array([(g + h) / 2 for g, h in zip(temp_foss, temp_foss1)])
            mins = [min(g, h) for g, h in zip(temp_foss, temp_foss1)]
            maxs = [max(g, h) for g, h in zip(temp_foss, temp_foss1)]
            std = np.array([np.std(np.array([g,h])) for g, h in zip(temp_foss, temp_foss1)])
            c, = ax.plot(temp_irs, means, label=demo, zorder=2)
            ax.scatter(temp_irs, means, c = c.get_color(), zorder=2)
            #ax.scatter(temp_irs, maxs, alpha=0.5, c=c.get_color())
            #ax.plot(temp_irs, maxs, alpha=0.5, c=c.get_color())
            ax.fill_between(temp_irs, means - std, means + std, alpha=0.2, zorder=2)
            #ax.errorbar(temp_irs, means, std, linestyle='None', marker='^', label=demo)

            ax1_irs.extend(temp_irs)

            i = 0
        elif i == 1 and dir[1]['shared_obs'] == 0 and dir[1]['shifted'] == 0:
            means = np.array([(g + h) / 2 for g, h in zip(temp_foss, temp_foss1)])
            mins = [min(g, h) for g, h in zip(temp_foss, temp_foss1)]
            maxs = [max(g, h) for g, h in zip(temp_foss, temp_foss1)]
            std = np.array([np.std(np.array([g,h])) for g, h in zip(temp_foss, temp_foss1)])
            c, = ax2.plot(temp_irs, means, label=demo, zorder=2)
            ax2.scatter(temp_irs, means, c = c.get_color(), zorder=2)
            #ax.scatter(temp_irs, maxs, alpha=0.5, c=c.get_color())
            #ax.plot(temp_irs, maxs, alpha=0.5, c=c.get_color())
            ax2.fill_between(temp_irs, means - std, means + std, alpha=0.2, zorder=2)
            #ax2.errorbar(temp_irs, means, std, linestyle='None', marker='o', label=demo, c=c.get_color())

            ax2_irs.extend(temp_irs)

            i = 0
        elif i == 1 and dir[1]['shared_obs'] == 0 and dir[1]['shifted'] == 1:
            means = np.array([(g + h) / 2 for g, h in zip(temp_foss, temp_foss1)])
            mins = [min(g, h) for g, h in zip(temp_foss, temp_foss1)]
            maxs = [max(g, h) for g, h in zip(temp_foss, temp_foss1)]
            std = np.array([np.std(np.array([g,h])) for g, h in zip(temp_foss, temp_foss1)])
            c, = ax3.plot(temp_irs, means, label=demo, zorder=2)
            ax3.scatter(temp_irs, means, c = c.get_color(), zorder=2)
            #ax.scatter(temp_irs, maxs, alpha=0.5, c=c.get_color())
            #ax.plot(temp_irs, maxs, alpha=0.5, c=c.get_color())
            ax3.fill_between(temp_irs, means - std, means + std, alpha=0.2, zorder=2)
            #ax2.errorbar(temp_irs, means, std, linestyle='None', marker='o', label=demo, c=c.get_color())

            ax3_irs.extend(temp_irs)

            i = 0
        else:
        #    ax.scatter(temp_irs, temp_foss, label=demo)
        #    prev, = ax.plot(temp_irs, temp_foss)
            i += 1

    ax.legend( loc='upper right', fontsize=17)
    ax.set_xscale('log', base=2)
    ax.set_ylim(0.91, 0.96)
    ax2.set_ylim(0.91, 0.96)
    ax2.legend( loc='upper right', fontsize=17)
    ax2.set_xscale('log', base=2)
    #ax3.set_ylim(0.91, 0.96)
    ax3.legend( loc='upper right', fontsize=17)
    ax3.set_xscale('log', base=2)

    from matplotlib.ticker import ScalarFormatter
    #for axis in [ax.xaxis, ax.yaxis]:
    #    axis.set_major_formatter(ScalarFormatter())
    #for axis in [ax2.xaxis, ax2.yaxis]:
    #    axis.set_major_formatter(FormatStrFormatter('%.4f'))

    ax.set_xticks(np.unique(ax1_irs))
    ax.set_xticklabels(np.unique(ax1_irs))
    ax2.set_xticks(np.unique(ax2_irs))
    ax2.set_xticklabels(np.unique(ax2_irs), rotation=45)
    ax3.set_xticks(np.unique(ax3_irs))
    ax3.set_xticklabels(np.unique(ax3_irs))
    ax.grid(zorder=1)
    ax2.grid(zorder=1)
    ax3.grid(zorder=1)

    #ax.get_xaxis().set_major_formatter(ScalarFormatter())
    #ax.set_xticklabels(np.unique(ax1_irs))
    #ax2.set_xticklabels(np.unique(ax2_irs))

    ax.axhline(BEST_SAC_VALUE, ls='--', lw=1, c='red', zorder=2)
    ax.axhline(BEST_SAC_VALUE - 0.005, ls='--', lw=1, c='grey', zorder=2)
    ax.axhline(BEST_SAC_VALUE + 0.005, ls='--', lw=1, c='grey', zorder=2)

    ax.text(0.001, BEST_SAC_VALUE - 0.0003, 'SAC Baseline', color='r', ha='left', va='top',
            transform=ax.get_yaxis_transform(), fontsize=17)
    ax2.axhline(BEST_SAC_VALUE, ls='--', lw=1, c='red', zorder=2)
    ax2.axhline(BEST_SAC_VALUE - 0.005, ls='--', lw=1, c='grey', zorder=2)
    ax2.axhline(BEST_SAC_VALUE + 0.005, ls='--', lw=1, c='grey', zorder=2)

    ax2.text(0.001, BEST_SAC_VALUE - 0.0003, 'SAC Baseline', color='r', ha='left', va='top',
            transform=ax.get_yaxis_transform(), fontsize=17)
    ax3.axhline(BEST_SAC_B3, ls='--', lw=1, c='red', zorder=2)
    ax3.axhline(BEST_SAC_B3 - 0.005, ls='--', lw=1, c='red', zorder=2)
    ax3.axhline(BEST_SAC_B3 + 0.005, ls='--', lw=1, c='red', zorder=2)
    ax3.axhline(BEST_SAC_B5, ls='--', lw=1, c='blue', zorder=2)
    ax3.axhline(BEST_SAC_B5 - 0.005, ls=':', lw=1, c='blue', zorder=2)
    ax3.axhline(BEST_SAC_B5 + 0.005, ls=':', lw=1, c='blue', zorder=2)

    ax.tick_params(axis='both', which='major', labelsize=17)
    ax2.tick_params(axis='both', which='major', labelsize=17)
    ax3.tick_params(axis='both', which='major', labelsize=17)

    #ax3.text(0.001, BEST_SAC_VALUE - 0.0003, 'SAC Baseline', color='r', ha='left', va='top',
    #         transform=ax.get_yaxis_transform(), fontsize=17)

    plt.tight_layout()
    plt.show()

    fig.savefig('results3.pdf')
    fig2.savefig('results4.pdf')
    fig3.savefig('results5.pdf')



if __name__ == '__main__':
    generate_sacdb2value()
