import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BEST_SAC_VALUE = 0.929
BEST_EVAL = 0.919
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
                            'demos': [2, 2, 2, 2, '2 (determ.)',
                                      4, 4, 4, 4,
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
                                           '8_determ_actions/demo_b16/no_extra_pol_update',
                                           '8_determ_actions/demo_b16/extra_pol_update',
                                           '9_interchanged_observations/demo_b6/no_extra_policy_update',
                                           '9_interchanged_observations/demo_b6/extra_policy_update',
                                           '9_interchanged_observations/demo_b6_determ/no_extra_pol_update',
                                           '9_interchanged_observations/demo_b6_determ/extra_pol_update',
                                           '3_demo_b5',
                                           '5_demo_b5_policyupdate',
                                           '7_shifted_demos/b5_demo_b5/non_extra_pol',
                                           '7_shifted_demos/b5_demo_b5/extra_pol',
                                           '7_shifted_demos/b5_demo_b6/non_extra_pol',
                                           '7_shifted_demos/b5_demo_b6/extra_pol',
                                           '7_shifted_demos/b3_demo_b3/non_extra_pol',
                                           '7_shifted_demos/b3_demo_b3/extra_pol',
                                           '7_shifted_demos/b3_demo_b6/non_extra_pol',
                                           '7_shifted_demos/b3_demo_b6/extra_pol',
                                           ],
                                 'demos': ['2 (shared obs.)', '2 (shared obs.)',
                                           '2 (shared obs, determ)', '2 (shared obs, determ)',
                                           2, 2,
                                           4, 4,
                                           'B6', 'B6',
                                           'B6 (determ.)', 'B6 (determ.)',
                                           'B16 (determ.)', 'B16 (determ.)',
                                           'B6 (shared obs.)', 'B6 (shared obs.)',
                                           'B6 (shared obs., determ)', 'B6 (shared obs., determ)',
                                           'B5', 'B5',
                                           'B5 (only B5s)', 'B5 (only B5s)',
                                           'B6 (only B5s)', 'B6 (only B5s)',
                                           'B3 (only B3s)', 'B3 (only B3s)',
                                           'B6 (only B3s)', 'B6 (only B3s)'],
                                 'extra_pols': [0, 1,  # 2 shared
                                                0, 1,  # 2 shared determ
                                                0, 1,  # 2
                                                0, 1,  # 4
                                                0, 1,  # B6
                                                0, 1,  # B6 determ
                                                0, 1,  # B16 determ
                                                0, 1,  # B6 shared
                                                0, 1,  # B6 shared determ
                                                0, 1,  # B5
                                                0, 1,  # B5 only B5s
                                                0, 1,  # B6 only B5s
                                                0, 1,
                                                0, 1
                                                ]})


eval_dirs = pd.DataFrame({'paths': ['Demo_B5/extra_pol',
                                    'Demo_B5/no_extra_pol',
                                    'Demo_B6/extra_pol',
                                    'Demo_B6/no_extra_pol',
                                    'Demo_B11/extra_pol',
                                    'Demo_B11/no_extra_pol',
                                    ],
                                 'demos': ['B5', 'B5',
                                           'B6', 'B6',
                                           'B11', 'B11',
                                           ],
                                 'extra_pols': [1, 0,
                                                1, 0,
                                                1, 0,
                                                ]})

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
                                    '2_own_reward/fossil_abs',
                                    '2_own_reward/fossil_share',
                                    '2_own_reward/tolovski',
                                    '2_own_reward/pricesolar/info_sharing',
                                    '2_own_reward/pricesolar/no_info_sharing',
                                    '2_own_reward_share_obs/own_marl/info_sharing',
                                    '2_own_reward_share_obs/own_marl/no_info_sharing',
                                    '2_own_reward_share_obs/own_marl2',
                                    '2_own_reward_share_obs/own_marl3',
                                    '2_own_reward_share_obs/own_marl4/info_sharing',
                                    '2_own_reward_share_obs/own_marl4/no_info_sharing',
                                    '2_own_reward_share_obs/fossil_abs/info_sharing',
                                    '2_own_reward_share_obs/fossil_abs/no_info_sharing',
                                    '2_own_reward_share_obs/fossil_share/info_sharing',
                                    '2_own_reward_share_obs/fossil_share/no_info_sharing',
                                    '2_own_reward_share_obs/tolovski/info_sharing',
                                    '2_own_reward_share_obs/tolovski/no_info_sharing',
                                    '2_own_reward/pricesolar/info_sharing',
                                    '2_own_reward/pricesolar/no_info_sharing',
                                    '3_add_renew_prod/no_obs_sharing/info_sharing',
                                    '3_add_renew_prod/no_obs_sharing/no_info_sharing',
                                    '3_add_renew_prod/obs_sharing/info_sharing',
                                    '3_add_renew_prod/obs_sharing/no_info_sharing',
                                    ],
                          'reward': ['classic', 'classic', 'classic', 'classic',
                                     'Own1', 'Own1', 'Own2', 'Own3', 'Own4', 'Own4', 'FossilAbs', 'FossilShare', 'Tolovski', 'PriceSolar', 'PriceSolar',
                                     'Own1', 'Own1', 'Own2', 'Own3', 'Own4', 'Own4', 'FossilAbs', 'FossilAbs', 'FossilShare', 'FossilShare', 'Tolovski', 'Tolovski', 'PriceSolar', 'PriceSolar',
                                     'Own4 (renew. prod.)', 'Own4 (renew. prod.)', 'Own4 (renew. prod.)',
                                     'Own4 (renew. prod.)', ],
                          'info_sharing': ['Yes', 'No', 'Yes', 'No',
                                           'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'No',
                                           'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No',
                                           'Yes', 'No', 'Yes', 'No', ],
                          'shared_observations': ['Yes', 'Yes', 'No', 'No',
                                                  'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No',
                                                  'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes',
                                                  'No', 'No', 'Yes', 'Yes', ]})

mode = 'sacdb234r'
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
    markers = {0.01: 'o',
               0.2: 'X',
               1.0: 's',
               1.5: '+'}
    demo_pos = {'2 (determ.)': 1,
                2: 2,
                4: 3,
                'B5': 4,
                'B6': 5}

    final_df = pd.DataFrame({'irs': irs,
                             'demos': demos,
                             'modes': modes,
                             'fossil_energy_consumptions': fossil_energy_consumptions})

    fig, ax = plt.subplots(figsize=(10, 7))

    if var == 1:
        for i, v in enumerate(final_df['fossil_energy_consumptions']):
            ax.scatter(demo_pos[demos[i]], v, color=colors[modes[i]], marker=markers[irs[i]], alpha=0.3)
        xticks = ax.get_xticks()
    else:
        ax = sns.scatterplot(x=final_df['demos'].map(demo_pos),
                             y='fossil_energy_consumptions',
                             hue=final_df['modes'].map(colors),
                             data=final_df,
                             style="irs",
                             zorder=4)
        for points in ax.collections:
            vertices = points.get_offsets().data
            if len(vertices) > 0:
                vertices[:, 0] += np.random.uniform(-0.35, 0.35, vertices.shape[0])
                points.set_offsets(vertices)
        xticks = ax.get_xticks()
        ax.set_xlim(xticks[0] - 0.5, xticks[-1] + 0.5)  # the limits need to be moved to show all the jittered dots
        # sns.move_legend(ax, bbox_to_anchor=(1.01, 1.02), loc='upper left')  # needs seaborn 0.11.2
        # sns.despine()

    ax.axhline(BEST_SAC_VALUE, ls='--', lw=1, c='red')
    ax.axhline(BEST_SAC_VALUE - 0.005, ls='--', lw=1, c='grey')
    ax.axhline(BEST_SAC_VALUE + 0.005, ls='--', lw=1, c='grey')

    ax.set_xticks(list(demo_pos.values()))
    ax.set_xticklabels(list(demo_pos.keys()))

    for t in range(len(xticks)):
        if t % 2 == 1:
            ax.axvline(xticks[t] - 0.5, c='#E6E6E6', lw=0.5)
            ax.axvline(xticks[t] + 0.5, c='#E6E6E6', lw=0.5)

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]

    handles = [f("s", colors[i]) for i in colors.keys()]
    handles += [f(markers[i], "k") for i in markers.keys()]

    labels = [f'Mode {i}' for i in colors.keys()] + ["ir0.01", "ir0.2", "ir1", "ir1.5"]

    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('SACDB2')


def generate_marl():
    dirs = marl_dirs

    rewards = []
    info_sharings = []
    shared_obs = []

    for dir in dirs.iterrows():
        reward = dir[1]['reward']
        info_sharing = dir[1]['info_sharing']
        shared_ob = dir[1]['shared_observations']

        try:
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
        except:
            pass

    colors = {'Yes': 'tab:blue',
              'No': 'tab:orange', }
    markers = {'Yes': 'o', 'No': 'X'}
    reward_pos = {'classic': 1,
                  'PriceSolar': 2,
                  'Own1': 3,
                  'Own2': 4,
                  'Own3': 5,
                  'Own4': 6,
                  'Own4 (renew. prod.)': 7,
                  'FossilAbs': 8,
                  'FossilShare': 9,
                  'Tolovski': 10}

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

        for ir in [0.0001, 0.001, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.8]:
            try:
                file = glob.glob(f'../experiments/SAC_DB2Value/{dir[1]["paths"]}/ir{ir}/kpis_*.csv')[0]
                kpis = pd.read_csv(file)
                kpis = kpis.set_index('kpi')
                kpis = kpis[(kpis['env_id'] == 'SAC_DB2Value Best') & (kpis['level'] == 'district')]
                v = round(kpis.loc['fossil_energy_consumption', 'net_value'] /
                          kpis.loc['fossil_energy_consumption', 'net_value_without_storage'], 3)

                irs.append(ir)
                extra_pols.append(extra_pol)
                demos.append(demo)
                fossil_energy_consumptions.append(v)
            except:
                print('Missing')

    colors = {0.0001: 'tab:blue',
              0.001: 'tab:orange',
              0.01: 'tab:green',
              0.03: 'tab:red',
              0.05: 'tab:purple',
              0.1: 'tab:brown',
              0.15: 'tab:pink',
              0.2: 'tab:grey',
              0.4: 'yellow',
              0.6: 'black',
              0.8: 'magenta'}
    markers = {0: 'o', 1: 'X'}
    demo_pos = {'2 (shared obs.)': 1,
                '2 (shared obs, determ)': 2,
                2: 3,
                4: 4,
                'B5': 5,
                'B6': 6,
                'B6 (determ.)': 7,
                'B6 (shared obs.)': 8,
                'B6 (shared obs., determ)': 9,
                'B16 (determ.)': 10,
                'B5 (only B5s)': 11,
                'B6 (only B5s)': 12,
                'B3 (only B3s)': 13,
                'B6 (only B3s)': 14}

    final_df = pd.DataFrame({'irs': irs,
                             'demos': demos,
                             'fossil_energy_consumptions': fossil_energy_consumptions,
                             'extra_pols': extra_pols})

    fig, ax = plt.subplots(figsize=(14, 7))
    final_df['irs'] = final_df['irs'].astype(str)

    if var == 1:
        for i, v in enumerate(final_df['fossil_energy_consumptions']):
            ax.scatter(demo_pos[demos[i]], v, color=colors[irs[i]], marker=markers[extra_pols[i]], alpha=0.5)
    else:
        ax = sns.scatterplot(x=final_df['demos'].map(demo_pos),
                             y='fossil_energy_consumptions',
                             hue=final_df['irs'].values,  # .map(colors),
                             data=final_df,
                             style='extra_pols',
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

    ax.set_xticks(list(demo_pos.values()))
    ax.set_xticklabels(list(demo_pos.keys()), rotation=90)

    for t in range(len(xticks)):
        if t % 2 == 1:
            ax.axvline(xticks[t] - 0.5, c='#E6E6E6', lw=0.5)
            ax.axvline(xticks[t] + 0.5, c='#E6E6E6', lw=0.5)

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]

    handles = [f("s", colors[i]) for i in colors.keys()]
    handles += [f(markers[i], "k") for i in markers.keys()]

    labels = [f'ir {i}' for i in colors.keys()] + ["No extra policy update", "Extra policy update"]

    # plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('SACDB2 Value')


def generate_eval():
    dirs = eval_dirs

    for dir in dirs.iterrows():
        demo = dir[1]['demos']
        extra_pol = dir[1]['extra_pols']

        for ir in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
            try:
                file = glob.glob(f'../experiments/New_Buildings/{dir[1]["paths"]}/ir{ir}/kpis_*.csv')[0]
                kpis = pd.read_csv(file)
                kpis = kpis.set_index('kpi')
                kpis = kpis[(kpis['env_id'] == 'SAC_DB2Value Best') & (kpis['level'] == 'district')]
                v = round(kpis.loc['fossil_energy_consumption', 'net_value'] /
                          kpis.loc['fossil_energy_consumption', 'net_value_without_storage'], 3)

                irs.append(ir)
                extra_pols.append(extra_pol)
                demos.append(demo)
                fossil_energy_consumptions.append(v)
            except:
                print('Missing')

    colors = {0.0001: 'tab:blue',
              0.001: 'tab:orange',
              0.01: 'tab:green',
              0.03: 'tab:red',
              0.05: 'tab:purple',
              0.1: 'tab:brown',
              0.15: 'tab:pink',
              0.2: 'tab:grey',
              0.4: 'yellow',
              0.6: 'black',
              0.8: 'magenta'}
    markers = {0: 'o', 1: 'X'}
    demo_pos = {'B5': 1,
                'B6': 2,
                'B11': 3}

    final_df = pd.DataFrame({'irs': irs,
                             'demos': demos,
                             'fossil_energy_consumptions': fossil_energy_consumptions,
                             'extra_pols': extra_pols})

    fig, ax = plt.subplots(figsize=(14, 7))
    final_df['irs'] = final_df['irs'].astype(str)

    if var == 1:
        for i, v in enumerate(final_df['fossil_energy_consumptions']):
            ax.scatter(demo_pos[demos[i]], v, color=colors[irs[i]], marker=markers[extra_pols[i]], alpha=0.5)
    else:
        ax = sns.scatterplot(x=final_df['demos'].map(demo_pos),
                             y='fossil_energy_consumptions',
                             hue=final_df['irs'].values,  # .map(colors),
                             data=final_df,
                             style='extra_pols',
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

    ax.axhline(BEST_EVAL, ls='--', lw=1, c='red')
    ax.axhline(BEST_EVAL - 0.005, ls='--', lw=1, c='grey')
    ax.axhline(BEST_EVAL + 0.005, ls='--', lw=1, c='grey')

    ax.set_xticks(list(demo_pos.values()))
    ax.set_xticklabels(list(demo_pos.keys()), rotation=90)

    for t in range(len(xticks)):
        if t % 2 == 1:
            ax.axvline(xticks[t] - 0.5, c='#E6E6E6', lw=0.5)
            ax.axvline(xticks[t] + 0.5, c='#E6E6E6', lw=0.5)

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]

    handles = [f("s", colors[i]) for i in colors.keys()]
    handles += [f(markers[i], "k") for i in markers.keys()]

    labels = [f'ir {i}' for i in colors.keys()] + ["No extra policy update", "Extra policy update"]

    # plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Evaluation')


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
    elif mode == 'eval':
        generate_eval()
    else:
        generate_sacdb2value()

    plt.tight_layout()
    plt.show()
