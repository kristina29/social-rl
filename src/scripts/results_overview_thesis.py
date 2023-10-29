import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BEST_SAC_VALUE = 0.929
Y_LIM = [0.91, 1.005]
sacdb2_dirs = pd.DataFrame({'paths': ['31_randomdemo/d2/ir0.01',
                                      '31_randomdemo/d2/ir0.2',
                                      '31_randomdemo/d2/ir1',
                                      '31_randomdemo/d2/ir1.5',
                                      '31_randomdemo/d2/det_ir0.2',
                                      '31_randomdemo/d4/ir0.01',
                                      '31_randomdemo/d4/ir0.2',
                                      '31_randomdemo/d4/ir1',
                                      '31_randomdemo/d4/ir1.5',
                                      '32_demo_b5/ir0.01',
                                      '32_demo_b5/ir0.2',
                                      '32_demo_b5/ir1',
                                      '32_demo_b5/ir1.5',
                                      '32_demo_b6/ir0.01',
                                      '32_demo_b6/ir0.2',
                                      '32_demo_b6/ir1',
                                      '32_demo_b6/ir1.5',
                                      ],
                            'demos': ['2 random', '2 random', '2 random', '2 random', '2 random\n(determ.)',
                                      '4 random', '4 random', '4 random', '4 random',
                                      'B5', 'B5', 'B5', 'B5',
                                      'B6', 'B6', 'B6', 'B6'],
                            'ir': [0.01, 0.2, 1, 1.5, 0.2,
                                   0.01, 0.2, 1, 1.5,
                                   0.01, 0.2, 1, 1.5,
                                   0.01, 0.2, 1, 1.5],
                            })

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
                                           ],
                                 'demos': ['2 random\n(shared obs.)', '2 random\n(shared obs.)',
                                           '2 random\n(shared obs.,\ndeterm.)', '2 random\n(shared obs.,\ndeterm.)',
                                           '2 random', '2 random',
                                           '4 random', '4 random',
                                           'B6', 'B6',
                                           'B6 (determ.)', 'B6 (determ.)',
                                           'B5', 'B5',],
                                 'extra_pols': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})

marl_dirs = pd.DataFrame({'paths': ['1_marlisa_classic/with_shared_obs/with_info_sharing',
                                    '1_marlisa_classic/with_shared_obs/without_info_sharing',
                                    '1_marlisa_classic/without_shared_obs/with_info_sharing',
                                    '1_marlisa_classic/without_shared_obs/without_info_sharing',
                                    '2_own_reward/own_marl/info_sharing',
                                    '2_own_reward/own_marl/no_info_sharing',
                                    '2_own_reward/own_marl2',
                                    '2_own_reward/own_marl3',
                                    '2_own_reward/own_marl4/info_sharing',
                                    '2_own_reward/own_marl4/no_info_sharing',
                                    '2_own_reward/pricesolar/info_sharing',
                                    '2_own_reward/pricesolar/no_info_sharing',
                                    '2_own_reward_share_obs/own_marl/info_sharing',
                                    '2_own_reward_share_obs/own_marl/no_info_sharing',
                                    '2_own_reward_share_obs/own_marl2',
                                    '2_own_reward_share_obs/own_marl3',
                                    '2_own_reward_share_obs/own_marl4/info_sharing',
                                    '2_own_reward_share_obs/own_marl4/no_info_sharing',
                                    '2_own_reward/pricesolar/info_sharing',
                                    '2_own_reward/pricesolar/no_info_sharing',
                                    '3_add_renew_prod/no_obs_sharing/info_sharing',
                                    '3_add_renew_prod/no_obs_sharing/no_info_sharing',
                                    '3_add_renew_prod/obs_sharing/info_sharing',
                                    '3_add_renew_prod/obs_sharing/no_info_sharing',
                                    ],
                          'reward': ['classic', 'classic', 'classic', 'classic',
                                     'Own1', 'Own1', 'Own2', 'Own3', 'Own4', 'Own4', 'PriceSolar', 'PriceSolar',
                                     'Own1', 'Own1', 'Own2', 'Own3', 'Own4', 'Own4', 'PriceSolar', 'PriceSolar',
                                     'Own4 (renew. prod.)', 'Own4 (renew. prod.)', 'Own4 (renew. prod.)',
                                     'Own4 (renew. prod.)', ],
                          'info_sharing': ['Yes', 'No', 'Yes', 'No',
                                           'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No',
                                           'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No',
                                           'Yes', 'No', 'Yes', 'No', ],
                          'shared_observations': ['Yes', 'Yes', 'No', 'No',
                                                  'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No',
                                                  'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes',
                                                  'No', 'No', 'Yes', 'Yes', ]})

mode = 'sacdb2v'
var = 2


def get_data_sacdb2(dirs):
    for dir in dirs.iterrows():
        ir = dir[1]['ir']
        demo = dir[1]['demos']

        for m in range(1, 7):
            try:
                file = glob.glob(f'../experiments/SAC_DB2/{dir[1]["paths"]}/socialMode{m}/kpis_*.csv')[0]
                kpis = pd.read_csv(file)
                kpis = kpis.set_index('kpi')
                kpis = kpis[(kpis['env_id'] == 'SAC_DB2 Best') & (kpis['level'] == 'district')]
                v = round(kpis.loc['fossil_energy_consumption', 'net_value'] /
                          kpis.loc['fossil_energy_consumption', 'net_value_without_storage'], 3)

                irs.append(ir)
                demos.append(demo)
                modes.append(m)
                fossil_energy_consumptions.append(v)
            except:
                pass

    return irs, demos, modes, fossil_energy_consumptions


def generate_sacdb2():
    dirs = sacdb2_dirs

    irs, demos, modes, fossil_energy_consumptions = get_data_sacdb2(dirs)

    colors = {1: 'tab:blue',
              2: 'tab:orange',
              3: 'tab:green',
              4: 'tab:red',
              5: 'tab:purple',
              6: 'tab:brown'}
    demo_pos = {'2 random\n(determ.)': 1,
                '2 random': 2,
                '4 random': 3,
                'B5': 4,
                'B6': 5}

    final_df = pd.DataFrame({'irs': irs,
                             'demos': demos,
                             'modes': modes,
                             'fossil_energy_consumptions': fossil_energy_consumptions})

    fig, ax = plt.subplots(figsize=(15, 10))

    ax = sns.scatterplot(x=final_df['demos'].map(demo_pos),
                         y='fossil_energy_consumptions',
                         hue=final_df['modes'].map(colors),
                         data=final_df,
                         zorder=4,
                         s=150)
    for points in ax.collections:
        vertices = points.get_offsets().data
        if len(vertices) > 0:
            vertices[:, 0] += np.random.uniform(-0.35, 0.35, vertices.shape[0])
            points.set_offsets(vertices)
    xticks = ax.get_xticks()
    ax.set_xlim(xticks[0] - 0.2, xticks[-1] + 0.2)  # the limits need to be moved to show all the jittered dots
    ax.set_ylim(Y_LIM[0], Y_LIM[1])

    ax.axhline(BEST_SAC_VALUE, ls='--', lw=1, c='red', zorder=2)
    ax.axhline(BEST_SAC_VALUE - 0.005, ls='--', lw=1, c='grey', zorder=2)
    ax.axhline(BEST_SAC_VALUE + 0.005, ls='--', lw=1, c='grey', zorder=2)

    ax.text(0.001, BEST_SAC_VALUE-0.0003, 'SAC Baseline', color='r', ha='left', va='top',
            transform=ax.get_yaxis_transform(), fontsize=17)

    ax.set_xticks(list(demo_pos.values()))
    ax.set_xticklabels(list(demo_pos.keys()), fontsize=17)

    ax.tick_params(axis='both', which='major', labelsize=17)

    ax.set_ylabel('KPI $fossil\_energy\_consumption$', fontsize=19)
    ax.set_xlabel('Demonstrators', fontsize=19)

    for t in range(len(xticks)):
        if t % 2 == 1:
            ax.axvline(xticks[t] - 0.5, c='#E6E6E6', lw=1, zorder=1)
            ax.axvline(xticks[t] + 0.5, c='#E6E6E6', lw=1, zorder=1)

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]

    handles = [f("o", colors[i]) for i in colors.keys()]

    labels = [f'Mode {i}' for i in colors.keys()]

    legend = plt.legend(handles, labels, #bbox_to_anchor=(1.05, 1),
                 loc='upper right', fontsize=21, markerscale=2)

    plt.tight_layout()

    fig.savefig('results.pdf')


def generate_marl():
    dirs = marl_dirs

    rewards = []
    info_sharings = []
    shared_obs = []

    for dir in dirs.iterrows():
        reward = dir[1]['reward']
        info_sharing = dir[1]['info_sharing']
        shared_ob = dir[1]['shared_observations']

        file = glob.glob(f'../experiments/SAC Others/{dir[1]["paths"]}/kpis_*.csv')[0]
        kpis = pd.read_csv(file)
        kpis = kpis.set_index('kpi')
        kpis = kpis[(kpis['env_id'] == 'MARLISA Best') & (kpis['level'] == 'district')]
        v = round(kpis.loc['fossil_energy_consumption', 'net_value'] /
                  kpis.loc['fossil_energy_consumption', 'net_value_without_storage'], 3)

        rewards.append(reward)
        info_sharings.append(info_sharing)
        shared_obs.append(shared_ob)
        fossil_energy_consumptions.append(v)

    colors = {'Yes': 'tab:blue',
              'No': 'tab:orange', }
    markers = {'Yes': 'o', 'No': 'X'}
    reward_pos = {'classic': 1,
                  'PriceSolar': 2,
                  'Own1': 3,
                  'Own2': 4,
                  'Own3': 5,
                  'Own4': 6,
                  'Own4 (renew. prod.)': 7}

    final_df = pd.DataFrame({'rewards': rewards,
                             'info_sharings': info_sharings,
                             'fossil_energy_consumptions': fossil_energy_consumptions,
                             'shared_obs': shared_obs})

    fig, ax = plt.subplots(figsize=(14, 7))

    if var == 1:
        for i, v in enumerate(final_df['fossil_energy_consumptions']):
            ax.scatter(reward_pos[rewards[i]], v, color=colors[shared_obs[i]], marker=markers[info_sharings[i]],
                       alpha=0.5)
    else:
        ax = sns.scatterplot(x=final_df['rewards'].map(reward_pos),
                             y='fossil_energy_consumptions',
                             hue=final_df['shared_obs'].values,  # .map(colors),
                             data=final_df,
                             style='info_sharings',
                             zorder=4)
        for points in ax.collections:
            vertices = points.get_offsets().data
            if len(vertices) > 0:
                vertices[:, 0] += np.random.uniform(-0.35, 0.35, vertices.shape[0])
                points.set_offsets(vertices)
        xticks = ax.get_xticks()
        ax.set_xlim(xticks[0] - 0.5, xticks[-1] + 0.5)  # the limits need to be moved to show all the jittered dots
        sns.move_legend(ax, bbox_to_anchor=(1.01, 1.02), loc='upper left')  # needs seaborn 0.11.2
        sns.despine()

    ax.axhline(BEST_SAC_VALUE, ls='--', lw=1, c='red')
    ax.axhline(BEST_SAC_VALUE - 0.005, ls='--', lw=1, c='grey')
    ax.axhline(BEST_SAC_VALUE + 0.005, ls='--', lw=1, c='grey')

    ax.set_xticks(list(reward_pos.values()))
    ax.set_xticklabels(list(reward_pos.keys()), rotation=90)

    for t in range(len(xticks)):
        if t % 2 == 1:
            ax.axvline(xticks[t] - 0.5, c='#E6E6E6', lw=0.5)
            ax.axvline(xticks[t] + 0.5, c='#E6E6E6', lw=0.5)

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]

    handles = [f("s", colors[i]) for i in colors.keys()]
    handles += [f(markers[i], "k") for i in markers.keys()]

    labels = [f'Shared observations: {i}' for i in colors.keys()] + ["Information Sharing: Yes",
                                                                     "Information Sharing: No"]

    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('MARLISA')


def generate_sacdb2value():
    dirs = sacdb2value_dirs

    for dir in dirs.iterrows():
        demo = dir[1]['demos']
        extra_pol = dir[1]['extra_pols']

        for ir in [0.0001, 0.001, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.4, 0.6, 0.8]:
        #for ir in [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.4]:#, 0.6, 0.8]:
            try:
                file = glob.glob(f'../experiments/SAC_DB2Value/{dir[1]["paths"]}/ir{ir}/kpis_*.csv')[0]
                kpis = pd.read_csv(file)
                kpis = kpis.set_index('kpi')
                kpis = kpis[(kpis['env_id'] == 'SAC_DB2Value Best') & (kpis['level'] == 'district')]
                v = round(kpis.loc['fossil_energy_consumption', 'net_value'] /
                          kpis.loc['fossil_energy_consumption', 'net_value_without_storage'], 3)

                if ir <= 0.01:
                    irs.append('$<=$ 0.01')
                elif ir >= 0.4:
                    irs.append('$>=$ 0.4')
                else:
                    irs.append(ir)
                extra_pols.append(extra_pol)
                demos.append(demo)
                fossil_energy_consumptions.append(v)
            except:
                print('Missing')

    colors = {#0.0001: 'tab:blue',
              #0.001: 'tab:orange',
              r'$\leq$0.01': 'tab:pink',
              0.03: 'tab:blue',
              0.05: 'tab:orange',
              0.1: 'tab:green',
              0.15: 'tab:red',
              0.2: 'tab:purple',
              r'$\geq$ 0.4': 'tab:grey',
              #0.6: 'black',
              #0.8: 'magenta'
            }
    markers = {0: 'o', 1: 'X'}
    demo_pos = {'2 random\n(shared obs.)': 1,
                '2 random\n(shared obs.,\ndeterm.)': 2,
                '2 random': 3,
                '4 random': 4,
                'B5': 5,
                'B6': 6,
                'B6 (determ.)': 7}

    final_df = pd.DataFrame({'irs': irs,
                             'demos': demos,
                             'fossil_energy_consumptions': fossil_energy_consumptions,
                             'extra_pols': extra_pols})

    fig, ax = plt.subplots(figsize=(15, 10))
    final_df['irs'] = final_df['irs'].astype(str)

    ax = sns.scatterplot(x=final_df['demos'].map(demo_pos),
                         y='fossil_energy_consumptions',
                         hue=final_df['irs'].values,  # .map(colors),
                         data=final_df,
                         zorder=4,
                         s=150)
    for points in ax.collections:
        vertices = points.get_offsets().data
        if len(vertices) > 0:
            vertices[:, 0] += np.random.uniform(-0.35, 0.35, vertices.shape[0])
            points.set_offsets(vertices)
    xticks = ax.get_xticks()
    ax.set_xlim(xticks[0] - 0.01, xticks[-1] + 0.01)  # the limits need to be moved to show all the jittered dots
    ax.set_ylim(Y_LIM[0], Y_LIM[1])

    #sns.move_legend(ax, loc='upper right')  # needs seaborn 0.11.2
    #sns.despine()

    ax.axhline(BEST_SAC_VALUE, ls='--', lw=1, c='red', zorder=2)
    ax.axhline(BEST_SAC_VALUE - 0.005, ls='--', lw=1, c='grey', zorder=2)
    ax.axhline(BEST_SAC_VALUE + 0.005, ls='--', lw=1, c='grey', zorder=2)

    ax.text(0.001, BEST_SAC_VALUE - 0.0003, 'SAC Baseline', color='r', ha='left', va='top',
            transform=ax.get_yaxis_transform(), fontsize=17)

    ax.set_xticks(list(demo_pos.values()))
    ax.set_xticklabels(list(demo_pos.keys()), rotation=0)

    ax.tick_params(axis='both', which='major', labelsize=17)

    ax.set_ylabel('KPI $fossil\_energy\_consumption$', fontsize=19)
    ax.set_xlabel('Demonstrators', fontsize=19)

    for t in range(len(xticks)):
        if t % 2 == 1:
            ax.axvline(xticks[t] - 0.5, c='#E6E6E6', lw=0.5, zorder=1)
            ax.axvline(xticks[t] + 0.5, c='#E6E6E6', lw=0.5, zorder=1)

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]

    handles = [f("o", colors[i]) for i in colors.keys()]

    labels = [i for i in colors.keys()]

    plt.legend(handles, labels, #bbox_to_anchor=(1.05, 1),
             loc='upper right', title='Imitation\nlearning rate', #r'$\alpha_i$',
               title_fontsize=19,
              fontsize=17, markerscale=2)
    plt.tight_layout()

    fig.savefig('results2.pdf')


if __name__ == '__main__':
    irs = []
    demos = []
    modes = []
    extra_pols = []
    fossil_energy_consumptions = []

    if mode == 'sacdb2':
        generate_sacdb2()
    elif mode == 'marl':
        generate_marl()
    else:
        generate_sacdb2value()

    plt.tight_layout()
    plt.show()
