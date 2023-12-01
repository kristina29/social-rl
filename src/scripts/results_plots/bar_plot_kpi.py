from collections import Mapping

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def plot_district_kpis1(kpis, mode) -> plt.Figure:
    kpis = kpis[kpis['level'] == 'district'].copy()

    kpi_names = {
        '1 - average_daily_renewable_share': 1,
        '1 - average_daily_renewable_share_grid': 2,
        '1 - used_pv_of_total_share': 3,
        'fossil_energy_consumption': 0
    }
    labels = [
        'fossil energy consumption',
        'avg. fossil share',
        'avg. fossil share grid',
        '1 - used pv'
    ]
    kpis = kpis[
        (kpis['kpi'].isin(kpi_names.keys()))
    ].dropna()

    # kpis['env_id'] = k
    kpis['rank'] = kpis['kpi'].map(kpi_names)
    kpis = kpis.drop(columns=['net_value', 'net_value_without_storage'])
    kpis = kpis.sort_values(by='rank')
    kpis = kpis.round(3)
    kpis = kpis[
        (kpis['env_id'].isin(['RBC', 'SAC Best']))
    ]
    kpis = kpis.replace('SAC Best', 'SAC Baseline')

    row_count = 1
    column_count = 1
    env_count = len(kpis)
    kpi_count = len(kpis['kpi'].unique())
    figsize = (16.0 * column_count, 0.125 * env_count * kpi_count * row_count)
    fig, ax = plt.subplots(row_count, column_count, figsize=figsize)

    width = 0.9

    sns.barplot(x='value', y='kpi', data=kpis, hue='env_id', ax=ax, width=width, zorder=2)
    #ax.axvline(1.0, color='black', linestyle='--', linewidth=1, label='')
    ax.grid(axis='x', zorder=0)
    ax.set_xlabel('KPI Value', fontsize=21)
    ax.set_ylabel(None)

    for s in ['right', 'top']:
        ax.spines[s].set_visible(False)

    for p in ax.patches:
        p.set_y(p.get_y() + (1 - width) * .5)

        ax.text(
            p.get_x() + p.get_width(),
            p.get_y() + p.get_height() / 2.0,
            p.get_width(), ha='left', va='center', fontsize=21
        )

    # print([x - (1-width) for x in ax.get_yticks()])
    ax.set_yticks([x + (1 - width) * 0.5 for x in ax.get_yticks()])
    ax.set_yticklabels(labels, fontsize=21)
    ax.tick_params(axis='both', which='major', labelsize=21)

    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=0.0, fontsize=21)
    plt.tight_layout()
    # fig.suptitle('KPIs at district-level', fontsize=16)

    if mode == 1:
        filename = 'sac_kpis'
    fig.savefig(f"{filename}.pdf")

    return fig


def plot_district_kpis_multiple(kpis, names, mode) -> plt.Figure:
    values = []
    kpi_ids = []
    env_ids = []

    kpi_names = {
        '1 - average_daily_renewable_share': 1,
        '1 - average_daily_renewable_share_grid': 2,
        '1 - used_pv_of_total_share': 3,
        'fossil_energy_consumption': 0
    }
    labels = [
        'fossil energy consumption',
        'avg. fossil share',
        'avg. fossil share grid',
        '1 - used pv'
    ]

    for i, kpi in enumerate(kpis):
        kpi = kpi[kpi['level'] == 'district'].copy()
        kpi = kpi[
            (kpi['kpi'].isin(kpi_names.keys()))
        ].dropna()

        kpi['rank'] = kpi['kpi'].map(kpi_names)

        if mode == 2:
            kpi = kpi[kpi['env_id'] == 'SAC']
        elif mode == 3:
            kpi = kpi[kpi['env_id'] == 'PRB_SAC']
        elif mode == 4:
            kpi = kpi[kpi['env_id'] == 'SAC Best']
        elif mode == 5:
            if i == 0:
                kpi = kpi[kpi['env_id'] == 'DDPG Best']
            else:
                kpi = kpi[kpi['env_id'] == 'PRB_DDPG']
        elif mode == 6:
            kpi = kpi[kpi['env_id'] == 'SAC Best']
        elif mode == 7:
            if i == 0 or i == 2:
                kpi = kpi[kpi['env_id'] == 'SAC Best']
            else:
                kpi = kpi[kpi['env_id'] == 'SAC_DB2Value Best']

        kpi['env_id'] = names[i]
        kpi = kpi.drop(columns=['net_value', 'net_value_without_storage'])
        kpi = kpi.sort_values(by='rank')

        values.extend(list(np.around(kpi['value'], 3)))
        kpi_ids.extend(list(kpi['kpi']))
        env_ids.extend(list(kpi['env_id']))

    kpis = pd.DataFrame({'value': values, 'kpi': kpi_ids, 'env_id': env_ids})

    row_count = 1
    column_count = 1
    env_count = len(kpis)
    kpi_count = len(kpis['kpi'].unique())

    if mode in [3, 6]:
        figsize = (18.0 * column_count, 0.2 * env_count * kpi_count * row_count)
    else:
        figsize = (18.0 * column_count, 0.125 * env_count * kpi_count * row_count)
    fig, ax = plt.subplots(row_count, column_count, figsize=figsize)

    if mode in [3, 6]:
        width = 0.7
    elif mode in [2, 5]:
        width = 0.75
    else:
        width = 0.9

    # if mode == 5:
    #    sns.barplot(x='value', y='kpi', data=kpis, hue='env_id', ax=ax,
    #            width=width , palette=['tab:blue','tab:green','tab:red','tab:purple'])
    # else:
    sns.barplot(x='value', y='kpi', data=kpis, hue='env_id', ax=ax, width=width, zorder=3)

    label = 'SAC Baseline'
    if mode in [2, 6]:
        label = 'SAC Baseline\n(Training)'

    if mode in [5]:
        ax.axvline(0.93, ymin=0.75, ymax=0.984, color='black', linestyle='--', linewidth=2, label=label,
                   zorder=4)
        ax.text(0.905, 0.978, '0.93', color='black', transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=17)
        ax.axvline(0.982, ymin=0.5, ymax=0.734, color='black', linestyle='--', linewidth=2, label='', zorder=4)
        ax.text(0.95, 0.728, '0.982', color='black', transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=17)
        ax.axvline(0.973, ymin=0.25, ymax=0.484, color='black', linestyle='--', linewidth=2, label='', zorder=4)
        ax.text(0.941, 0.478, '0.973', color='black', transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=17)
        ax.axvline(0.767, ymin=0, ymax=0.234, color='black', linestyle='--', linewidth=2, label='', zorder=4)
        ax.text(0.735, 0.228, '0.767', color='black', transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=17)
    if mode == 6:
        ax.axvline(0.93, ymin=0.75, ymax=0.969, color='black', linestyle='--', linewidth=2, label=label,
                   zorder=4)
        ax.text(0.905, 0.964, '0.93', color='black', transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=17)
        ax.axvline(0.982, ymin=0.5, ymax=0.719, color='black', linestyle='--', linewidth=2, label='', zorder=4)
        ax.text(0.95, 0.714, '0.982', color='black', transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=17)
        ax.axvline(0.973, ymin=0.25, ymax=0.469, color='black', linestyle='--', linewidth=2, label='', zorder=4)
        ax.text(0.941, 0.464, '0.973', color='black', transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=17)
        ax.axvline(0.767, ymin=0, ymax=0.219, color='black', linestyle='--', linewidth=2, label='', zorder=4)
        ax.text(0.735, 0.214, '0.767', color='black', transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=17)
    elif mode == 2:
        ax.axvline(0.93, ymin=0.77, ymax=1, color='black', linestyle='--', linewidth=2, label=label, zorder=4)
        ax.text(0.905, 0.998, '0.93', color='black', transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=17, zorder=4)
        ax.axvline(0.982, ymin=0.52, ymax=0.75, color='black', linestyle='--', linewidth=2, label='', zorder=4)
        ax.text(0.95, 0.748, '0.982', color='black', transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=17, zorder=4)
        ax.axvline(0.973, ymin=0.28, ymax=0.5, color='black', linestyle='--', linewidth=2, label='', zorder=4)
        ax.text(0.941, 0.497, '0.973', color='black', transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=17, zorder=4)
        ax.axvline(0.767, ymin=0.03, ymax=0.25, color='black', linestyle='--', linewidth=2, label='', zorder=4)
        ax.text(0.735, 0.247, '0.767', color='black', transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=17, zorder=4)
    elif mode in [3]:
        ax.axvline(0.93, ymin=0.77, ymax=1, color='black', linestyle='--', linewidth=2, label=label, zorder=4)
        ax.text(0.905, 0.998, '0.93', color='black', transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=17, zorder=4)
        ax.axvline(0.982, ymin=0.52, ymax=0.75, color='black', linestyle='--', linewidth=2, label='', zorder=4)
        ax.text(0.95, 0.748, '0.982', color='black', transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=17, zorder=4)
        ax.axvline(0.973, ymin=0.28, ymax=0.5, color='black', linestyle='--', linewidth=2, label='', zorder=4)
        ax.text(0.941, 0.497, '0.973', color='black', transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=17, zorder=4)
        ax.axvline(0.767, ymin=0.03, ymax=0.25, color='black', linestyle='--', linewidth=2, label='', zorder=4)
        ax.text(0.735, 0.247, '0.767', color='black', transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=17, zorder=4)
    #else:
        #ax.axvline(1, color='black', linestyle='--', linewidth=1, label='')

    if mode in [2, 3, 4, 7]:
        ax.grid(axis='x', zorder=0)

    ax.set_xlabel('KPI Value', fontsize=21)
    ax.set_ylabel(None)

    for s in ['right', 'top']:
        ax.spines[s].set_visible(False)

    if not mode in [2, 3]:
        i = 0
        for p in ax.patches:
            p.set_y(p.get_y() + (1 - width) * .5)

            if mode == 6 and i in [1,2,5,6]:
                if i in [1,5]:
                    ax.text(
                        0.988,
                        p.get_y() + p.get_height() / 2.0,
                        p.get_width(), ha='left', va='center', fontsize=21
                    )
                elif i in [2,6]:
                    ax.text(
                        0.981,
                        p.get_y() + p.get_height() / 2.0,
                        p.get_width(), ha='left', va='center', fontsize=21
                    )
            else:
                ax.text(
                    p.get_x() + p.get_width(),
                    p.get_y() + p.get_height() / 2.0,
                    p.get_width(), ha='left', va='center', fontsize=21
                )
            i += 1

    # print([x - (1-width) for x in ax.get_yticks()])
    if mode in [2,3]:
        ax.set_yticks([x for x in ax.get_yticks()])
    else:
        ax.set_yticks([x + (1 - width) * 0.5 for x in ax.get_yticks()])
    ax.set_yticklabels(labels, fontsize=23)
    ax.tick_params(axis='both', which='major', labelsize=21)

    # fig.suptitle('KPIs at district-level', fontsize=16)
    if mode == 4:
        val = 0.74
    else:
        val = 0.77

    plt.tight_layout(rect=[0, 0, val, 1])

    if mode == 2:
        val = 1.3
    elif mode == 3:
        val = 1.4
    elif mode == 4:
        val = 1.33
    elif mode == 5:
        val = 1.425
    elif mode == 6:
        val = 1.33
    else:
        val = 1.425

    ax.legend(loc='upper right', bbox_to_anchor=(val, 1.0), framealpha=0, fontsize=21)
    # plt.show()

    if mode == 2:
        filename = 'pretrained_kpis'
    if mode == 3:
        filename = 'prb_kpis'
    if mode == 4:
        filename = 'eval_kpis'
    if mode == 5:
        filename = 'prb_ddpg_kpis'
    if mode == 6:
        filename = 'shifted_sac_kpis'
    if mode == 7:
        filename = 'social_kpis'

    fig.savefig(f"{filename}.pdf")

    return fig


if __name__ == '__main__':
    mode = 4

    if mode == 1:
        kpis = pd.read_csv('../experiments/SAC/30_renewable_prod/reward_05pvprice/0.5/kpis_mean.csv'),
    elif mode == 2:
        kpis = [pd.read_csv('../experiments/SAC/15_b3_demonstrator/kpis_20231030T145351.csv'),
                pd.read_csv('../experiments/SAC/10_b5_demonstrator/kpis_20231002T132420.csv'),
                pd.read_csv('../experiments/SAC/09_b6_demonstrator/kpis_20231002T132428.csv'),
                pd.read_csv('../experiments/SAC/17_b11_demonstrator/kpis_20231113T143302.csv'),
                ]
        names = ['D3', 'D5', 'D6', 'D11']
    elif mode == 3:
        kpis = [  # pd.read_csv('../experiments/SAC/30_renewable_prod/reward_05pvprice/0.5/kpis_mean.csv'),
            pd.read_csv('../experiments/Imitation_Learning/demo_b5/kpis_20231026T121912.csv'),
            pd.read_csv('../experiments/Imitation_Learning/demo_b6/kpis_20231026T122547.csv'),
        ]
        names = [  # 'Baseline SAC',
            'Transitions of D5', 'Transitions of D6']
    elif mode == 4:
        kpis = [pd.read_csv('../experiments/SAC/30_renewable_prod/reward_05pvprice/0.5/kpis_mean.csv'),
                pd.read_csv('../experiments/Evaluation_Buildings/SAC Baseline/kpis_20231113T132158.csv'),
                ]
        names = ['SAC Baseline\n(Training)', 'SAC Baseline\n(Evaluation)', ]
    elif mode == 5:
        kpis = [  # pd.read_csv('../experiments/SAC/30_renewable_prod/reward_05pvprice/0.5/kpis_mean.csv'),
            pd.read_csv('../experiments/SAC/DDPG/kpis_20231107T143713.csv'),
            pd.read_csv('../experiments/Imitation_Learning/demo_b5_ddpg/kpis_20231106T170835.csv'),
            pd.read_csv('../experiments/Imitation_Learning/demo_b6_ddpg/kpis_20231106T170845.csv'),
        ]
        names = [  # 'Baseline SAC',
            'DDPG', 'DDPG with transitions\nof D5', 'DDPG with transitions\nof D6', ]
    elif mode == 6:
        kpis = [pd.read_csv('../experiments/SAC/16_shifted_buildings/shifted_b3/kpis_20231031T173335.csv'),
                pd.read_csv('../experiments/SAC/16_shifted_buildings/shifted_b5/kpis_20231031T194250.csv'),
                ]
        names = ['SAC Baseline\n(Shifted B3)', 'SAC Baseline\n(Shifted B5)']
    elif mode == 7:
        kpis = [pd.read_csv('../experiments/SAC/30_renewable_prod/reward_05pvprice/0.5/kpis_mean.csv'),
                pd.read_csv(
                    '../experiments/SAC-DemoQ/8_determ_actions/demo_b6/non_extra_pol_update/ir0.15/kpis_20231005T120640.csv'),
                pd.read_csv('../experiments/Evaluation_Buildings/SAC Baseline/kpis_20231113T132158.csv'),
                pd.read_csv('../experiments/Evaluation_Buildings/Demo_B6/extra_pol/ir0.25/kpis_20231113T164524.csv'),
                ]
        names = ['Training SAC Baseline', 'Training SAC-DemoQ',
                 'Evaluation SAC Baseline',
                 'Evaluation SAC-DemoQ',
                 ]

    if len(kpis) == 1:
        plot_district_kpis1(kpis[0], mode)
    else:
        plot_district_kpis_multiple(kpis, names, mode)

    # plt.show()
