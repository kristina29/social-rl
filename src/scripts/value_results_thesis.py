import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

BEST_SAC_VALUE = 0.93
BEST_SAC_B3 = 0.94
BEST_SAC_B5 = 0.957
Y_LIM = [0.91, 1.005]

BEST_EVAL = 0.919


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
                                 'demos': ['2 random', '2 random',
                                           '2 random (determ.)', '2 random (determ.)',
                                           '2 random', '2 random',
                                           '4 random', '4 random',
                                           'B6', 'B6',
                                           'B6 (determ.)', 'B6 (determ.)',
                                           'B5', 'B5',
                                           'B6', 'B6',
                                           'B6 (determ.)', 'B6 (determ.)',
                                           'B5 (Shifted B5)', 'B5 (Shifted B5)',
                                           'B6 (Shifted B5)', 'B6 (Shifted B5)',
                                           'B3 (Shifted B3)', 'B3 (Shifted B3)',
                                           'B6 (Shifted B3)', 'B6 (Shifted B3)'
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
                                             1, 1, 1, 1, 1, 1, 1, 1],
                                 'colors': [None, None, None, None,
                                            None, None, None, None,
                                            None, None, None, None,
                                            None, None,
                                            None, None, None, None,
                                            'lightskyblue', 'lightskyblue', 'blue', 'blue',
                                            'lawngreen', 'lawngreen', 'green', 'green', ]
                                 })

eval_dirs = pd.DataFrame({'paths': [#'Demo_B5/extra_pol',
                                    #'Demo_B5/no_extra_pol',
                                    'Demo_B6/extra_pol',
                                    'Demo_B6/no_extra_pol',
                                    'Demo_B11/extra_pol',
                                    'Demo_B11/no_extra_pol',
                                    ],
                                 'demos': [#'B5', 'B5',
                                           'B6 (determ.)', 'B6 (determ.)',
                                           'B11 (determ.)', 'B11 (determ.)',
                                           ],
                                 'extra_pols': [#1, 0,
                                                1, 0,
                                                1, 0,
                                                ]})



def generate_sacdb2value():
    dirs = sacdb2value_dirs

    fig, ax = plt.subplots(figsize=(15,7))
    fig2, ax2 = plt.subplots(figsize=(15,7))
    fig3, ax3 = plt.subplots(figsize=(15,7))

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
            std = np.array([np.std(np.array([g,h])) for g, h in zip(temp_foss, temp_foss1)])

            c, = ax.plot(temp_irs, means, label=demo, zorder=3)
            ax.scatter(temp_irs, means, c = c.get_color(), zorder=3)
            ax.fill_between(temp_irs, means - std, means + std, alpha=0.2, zorder=3)

            ax1_irs.extend(temp_irs)
            i = 0
        elif i == 1 and dir[1]['shared_obs'] == 0 and dir[1]['shifted'] == 0:
            means = np.array([(g + h) / 2 for g, h in zip(temp_foss, temp_foss1)])
            std = np.array([np.std(np.array([g,h])) for g, h in zip(temp_foss, temp_foss1)])
            c, = ax2.plot(temp_irs, means, label=demo, zorder=3)
            ax2.scatter(temp_irs, means, c = c.get_color(), zorder=3)
            ax2.fill_between(temp_irs, means - std, means + std, alpha=0.2, zorder=3)

            ax2_irs.extend(temp_irs)
            i = 0
        elif i == 1 and dir[1]['shared_obs'] == 0 and dir[1]['shifted'] == 1:
            c = dir[1]['colors']
            means = np.array([(g + h) / 2 for g, h in zip(temp_foss, temp_foss1)])
            std = np.array([np.std(np.array([g,h])) for g, h in zip(temp_foss, temp_foss1)])
            ax3.plot(temp_irs, means, label=demo, c=c, zorder=3)
            ax3.scatter(temp_irs, means, c=c, zorder=3)
            ax3.fill_between(temp_irs, means - std, means + std, color=c, alpha=0.2, zorder=3)

            ax3_irs.extend(temp_irs)

            i = 0
        else:
            i += 1

    for a in [ax, ax2, ax3]:
        a.set_xscale('log', base=2)
        a.set_ylim(0.91, 0.975)
        a.set_ylabel('KPI $fossil\_energy\_consumption$', fontsize=19)
        a.set_xlabel(r'Imitation learning rate $\alpha_i$', fontsize=19)
        a.grid(zorder=1)
        a.tick_params(axis='both', which='major', labelsize=17)

    ax.legend(loc='upper left', title=r'Demonstrator', title_fontsize=18, fontsize=17, ncol=2)
    ax2.legend( loc='upper left', title=r'Demonstrator', title_fontsize=18, fontsize=17, ncol=2)
    ax3.legend(loc='lower left', fontsize=17, title=r'Demonstrator (Building Data)', title_fontsize=18, ncol=2)

    ax.set_xticks(np.unique(ax1_irs))
    ax.set_xticklabels(np.unique(ax1_irs))
    ax2.set_xticks(np.unique(ax2_irs))
    ax2.set_xticklabels(np.unique(ax2_irs), rotation=45)
    ax3.set_xticks(np.unique(ax3_irs))
    ax3.set_xticklabels(np.unique(ax3_irs))

    ax.axhline(BEST_SAC_VALUE, ls='--', lw=1, c='red', zorder=2)
    ax.axhline(BEST_SAC_VALUE - 0.005, ls='--', lw=1, c='grey', zorder=2)
    ax.axhline(BEST_SAC_VALUE + 0.005, ls='--', lw=1, c='grey', zorder=2)
    ax.text(0.001, BEST_SAC_VALUE - 0.0003, 'SAC Baseline', color='r', ha='left', va='top',
             transform=ax.get_yaxis_transform(), fontsize=17)

    ax2.axhline(BEST_SAC_VALUE, ls='--', lw=1, c='red', zorder=2)
    ax2.axhline(BEST_SAC_VALUE - 0.005, ls='--', lw=1, c='grey', zorder=2)
    ax2.axhline(BEST_SAC_VALUE + 0.005, ls='--', lw=1, c='grey', zorder=2)
    ax2.text(0.001, BEST_SAC_VALUE - 0.0003, 'SAC Baseline', color='r', ha='left', va='top',
            transform=ax2.get_yaxis_transform(), fontsize=17)

    ax3.axhline(BEST_SAC_B3, ls='--', lw=1, c='tab:green', zorder=2)
    ax3.axhline(BEST_SAC_B3 - 0.005, ls=':', lw=1.5, c='tab:green', zorder=2)
    ax3.axhline(BEST_SAC_B3 + 0.005, ls=':', lw=1.5, c='tab:green', zorder=2)
    ax3.axhline(BEST_SAC_B5, ls='--', lw=1, c='tab:blue', zorder=2)
    ax3.axhline(BEST_SAC_B5 - 0.005, ls=':', lw=1.5, c='tab:blue', zorder=2)
    ax3.axhline(BEST_SAC_B5 + 0.005, ls=':', lw=1.5, c='tab:blue', zorder=2)
    ax3.text(0.001, BEST_SAC_B3 - 0.0003, 'SAC (Shifted B3)', color='tab:green', ha='left', va='top',
             transform=ax3.get_yaxis_transform(), fontsize=17)
    ax3.text(0.001, BEST_SAC_B5 - 0.0003, 'SAC (Shifted B5)', color='tab:blue', ha='left', va='top',
             transform=ax3.get_yaxis_transform(), fontsize=17)

    fig.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    plt.show()

    fig.savefig('results3.pdf')
    fig2.savefig('results4.pdf')
    fig3.savefig('results5.pdf')


def generate_eval():
    dirs = eval_dirs

    fig, ax = plt.subplots(figsize=(15,7))

    i = 0

    ax1_irs = []

    for dir in dirs.iterrows():
        demo = dir[1]['demos']
        extra_pol = dir[1]['extra_pols']

        if i==0:
            temp_irs = []
            temp_foss = []
            temp_foss1 = []

        for ir in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
            # for ir in [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.4]:#, 0.6, 0.8]:
            try:
                file = glob.glob(f'../experiments/New_Buildings/{dir[1]["paths"]}/ir{ir}/kpis_*.csv')[0]
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

        if i == 1:
            means = np.array([(g + h) / 2 for g, h in zip(temp_foss, temp_foss1)])
            std = np.array([np.std(np.array([g,h])) for g, h in zip(temp_foss, temp_foss1)])

            c, = ax.plot(temp_irs, means, label=demo, zorder=3)
            ax.scatter(temp_irs, means, c = c.get_color(), zorder=3)
            ax.fill_between(temp_irs, means - std, means + std, alpha=0.2, zorder=3)

            ax1_irs.extend(temp_irs)
            i = 0
        else:
            i += 1

    for a in [ax]:
        a.set_xscale('log', base=2)
        a.set_ylim(0.905, 0.955)
        a.set_ylabel('KPI $fossil\_energy\_consumption$', fontsize=19)
        a.set_xlabel(r'Imitation learning rate $\alpha_i$', fontsize=19)
        a.grid(zorder=1)
        a.tick_params(axis='both', which='major', labelsize=17)

    ax.legend(loc='upper left', title=r'Demonstrator', title_fontsize=18, fontsize=17, ncol=2)

    ax.set_xticks(np.unique(ax1_irs))
    ax.set_xticklabels(np.unique(ax1_irs))

    ax.axhline(BEST_EVAL, ls='--', lw=1, c='red', zorder=2)
    ax.axhline(BEST_EVAL - 0.005, ls='--', lw=1, c='grey', zorder=2)
    ax.axhline(BEST_EVAL + 0.005, ls='--', lw=1, c='grey', zorder=2)
    ax.text(0.001, BEST_EVAL - 0.0003, 'SAC', color='r', ha='left', va='top',
             transform=ax.get_yaxis_transform(), fontsize=17)


    fig.tight_layout()


    plt.show()

    fig.savefig('results.pdf')



if __name__ == '__main__':
    #generate_sacdb2value()
    generate_eval()
