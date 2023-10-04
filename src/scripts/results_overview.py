import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BEST_SAC_VALUE = 0.929
sacdb2_dirs = pd.DataFrame({'paths': ['31_randomdemo/d2/ir0.01',
                                      '31_randomdemo/d2/ir0.2',
                                      '31_randomdemo/d4/ir0.01',
                                      '31_randomdemo/d4/ir0.2',
                                      '32_demo_b5/ir0.01',
                                      '32_demo_b5/ir0.2',
                                      '32_demo_b5/ir1',
                                      '32_demo_b6/ir0.01',
                                      '32_demo_b6/ir0.2',
                                      '32_demo_b6/ir1',
                                      ],
                            'demos': [2, 2, 4, 4, 'B5', 'B5', 'B5', 'B6', 'B6', 'B6'],
                            'ir': [0.01, 0.2, 0.01, 0.2, 0.01, 0.2, 1, 0.01, 0.2, 1],
                            })

sacdb2value_dirs = pd.DataFrame({'paths': ['1_randomdemo/d2',
                                           '1_randomdemo/d2_extrapol',
                                           '1_randomdemo/d4',
                                           '1_randomdemo/d4_extrapol',
                                           '2_demo_b6',
                                           '4_demo_b6_policyupdate',
                                           '3_demo_b5',
                                           '5_demo_b5_policyupdate'
                                           ],
                                 'demos': [2, 2, 4, 4, 'B6', 'B6', 'B5', 'B5'],
                                 'extra_pols': [0, 1, 0, 1, 0, 1, 0, 1]})

mode = 'sacdb2'
var = 2


def get_data_sacdb2(dirs):
    for dir in dirs.iterrows():
        ir = dir[1]['ir']
        demo = dir[1]['demos']

        for m in range(1, 7):
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
               1.0: 's'}
    demo_pos = {2: 1,
                4: 2,
                'B5': 3,
                'B6': 4}

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

    labels = [f'Mode {i}' for i in colors.keys()] + ["ir0.01", "ir0.2", "ir1"]

    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('SACDB2')


def generate_sacdb2value():
    dirs = sacdb2value_dirs

    for dir in dirs.iterrows():
        demo = dir[1]['demos']
        extra_pol = dir[1]['extra_pols']

        for ir in [0.0001, 0.001, 0.01, 0.03, 0.05, 0.1]:
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

    colors = {0.0001: 'tab:blue',
              0.001: 'tab:orange',
              0.01: 'tab:green',
              0.03: 'tab:red',
              0.05: 'tab:purple',
              0.1: 'tab:brown'}
    markers = {0: 'o', 1: 'X'}
    demo_pos = {2: 1,
                4: 2,
                'B5': 3,
                'B6': 4}

    final_df = pd.DataFrame({'irs': irs,
                             'demos': demos,
                             'fossil_energy_consumptions': fossil_energy_consumptions,
                             'extra_pols': extra_pols})

    fig, ax = plt.subplots(figsize=(10, 7))

    if var == 1:
        for i, v in enumerate(final_df['fossil_energy_consumptions']):
            ax.scatter(demo_pos[demos[i]], v, color=colors[irs[i]], marker=markers[extra_pols[i]], alpha=0.5)
    else:
        ax = sns.scatterplot(x=final_df['demos'].map(demo_pos),
                             y='fossil_energy_consumptions',
                             hue=final_df['irs'].map(colors),
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
    ax.set_xticklabels(list(demo_pos.keys()))

    for t in range(len(xticks)):
        if t % 2 == 1:
            ax.axvline(xticks[t] - 0.5, c='#E6E6E6', lw=0.5)
            ax.axvline(xticks[t] + 0.5, c='#E6E6E6', lw=0.5)

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]

    handles = [f("s", colors[i]) for i in colors.keys()]
    handles += [f(markers[i], "k") for i in markers.keys()]

    labels = [f'ir {i}' for i in colors.keys()] + ["No extra policy update", "Extra policy update"]

    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('SACDB2 Value')


if __name__ == '__main__':
    irs = []
    demos = []
    modes = []
    extra_pols = []
    fossil_energy_consumptions = []

    if mode == 'sacdb2':
        generate_sacdb2()
    else:
        generate_sacdb2value()

    plt.tight_layout()
    plt.show()
