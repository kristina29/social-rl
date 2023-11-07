from collections import Mapping

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_district_kpis1(kpis) -> plt.Figure:
    kpis = kpis[kpis['level'] == 'district'].copy()

    kpi_names = {
        '1 - average_daily_renewable_share': 1,
        '1 - average_daily_renewable_share_grid': 2,
        '1 - used_pv_of_total_share': 3,
        'fossil_energy_consumption': 0
    }
    labels = [
        'fossil_energy_consumption',
        '1 - avg_daily_renewable_share',
        '1 - avg_daily_renewable_share_grid',
        '1 - used_pv_of_total_share'
    ]
    kpis = kpis[
        (kpis['kpi'].isin(kpi_names.keys()))
    ].dropna()

    # kpis['env_id'] = k
    kpis['rank'] = kpis['kpi'].map(kpi_names)
    kpis = kpis.drop(columns=['net_value', 'net_value_without_storage'])
    kpis = kpis.sort_values(by='rank')

    row_count = 1
    column_count = 1
    env_count = len(kpis)
    kpi_count = len(kpis['kpi'].unique())
    figsize = (16.0 * column_count, 0.125 * env_count * kpi_count * row_count)
    fig, ax = plt.subplots(row_count, column_count, figsize=figsize)

    width = 0.9

    sns.barplot(x='value', y='kpi', data=kpis, hue='env_id', ax=ax, width=width)
    ax.axvline(1.0, color='black', linestyle='--', linewidth=1, label='')
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })

    for s in ['right', 'top']:
        ax.spines[s].set_visible(False)

    for p in ax.patches:
        p.set_y(p.get_y() + (1-width) * .5)

        ax.text(
            p.get_x() + p.get_width(),
            p.get_y() + p.get_height() / 2.0,
            p.get_width(), ha='left', va='center', fontsize=21
        )

    #print([x - (1-width) for x in ax.get_yticks()])
    ax.set_yticks([x + (1 - width) * 0.5 for x in ax.get_yticks()])
    ax.set_yticklabels(labels, fontsize=21)
    ax.tick_params(axis='both', which='major', labelsize=21)

    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=0.0, fontsize=21)
    plt.tight_layout()
    #fig.suptitle('KPIs at district-level', fontsize=16)


    fig.savefig("kpis.pdf")

    return fig

def plot_district_kpis_multiple(kpis, names) -> plt.Figure:
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
        'fossil_energy_consumption',
        '1 - avg_daily_renewable_share',
        '1 - avg_daily_renewable_share_grid',
        '1 - used_pv_of_total_share'
    ]

    for i, kpi in enumerate(kpis):
        kpi = kpi[kpi['level'] == 'district'].copy()
        kpi = kpi[
            (kpi['kpi'].isin(kpi_names.keys()))
        ].dropna()

        kpi['rank'] = kpi['kpi'].map(kpi_names)

        if i==0:
            kpi = kpi[kpi['env_id'] == 'SAC Best']
            c = 'tab:blue'
        elif i==1:
            kpi = kpi[kpi['env_id'] == 'DDPG Best']
            c = 'tab:red'
        else:
            kpi = kpi[kpi['env_id'] == 'PRB_DDPG']
            if i==2:
                c = 'tab:orange'
            else:
                c = 'tab:green'

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
    figsize = (18.0 * column_count, 0.125 * env_count * kpi_count * row_count)
    fig, ax = plt.subplots(row_count, column_count, figsize=figsize)

    width = 0.9

    sns.barplot(x='value', y='kpi', data=kpis, hue='env_id', ax=ax, width=width, palette=['tab:blue','tab:red','tab:orange','tab:green'])
    ax.axvline(1.0, color='black', linestyle='--', linewidth=1, label='')
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })

    for s in ['right', 'top']:
        ax.spines[s].set_visible(False)

    for p in ax.patches:
        p.set_y(p.get_y() + (1-width) * .5)

        ax.text(
            p.get_x() + p.get_width(),
            p.get_y() + p.get_height() / 2.0,
            p.get_width(), ha='left', va='center', fontsize=21
        )

    #print([x - (1-width) for x in ax.get_yticks()])
    ax.set_yticks([x + (1 - width) * 0.5 for x in ax.get_yticks()])
    ax.set_yticklabels(labels, fontsize=21)
    ax.tick_params(axis='both', which='major', labelsize=21)

    #fig.suptitle('KPIs at district-level', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.72, 1])
    ax.legend(loc='upper right', bbox_to_anchor=(1.73, 1.0), framealpha=0, fontsize=21)
    #plt.show()
    fig.savefig("kpis.pdf")

    return fig


if __name__ == '__main__':
    kpis = [pd.read_csv('../experiments/SAC_DB2/30_renewable_prod/reward_05pvprice/0.5/kpis_mean.csv'),
            pd.read_csv('../experiments/SAC/DDPG/kpis_20231107T143713.csv'),
            pd.read_csv('../experiments/SAC_DB2/33_demo_replaybuffer/demo_b5_ddpg/kpis_20231106T170835.csv'),
            pd.read_csv('../experiments/SAC_DB2/33_demo_replaybuffer/demo_b6_ddpg/kpis_20231106T170845.csv'),]

    if len(kpis) == 1:
        plot_district_kpis1(kpis[0])
    else:
        plot_district_kpis_multiple(kpis, ['Baseline SAC', 'DDPG', 'DDPG with transitions of B5', 'DDPG with transitions of B6'])

    #plt.show()