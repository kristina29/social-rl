from collections import Mapping

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_district_kpis(kpis) -> plt.Figure:
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

    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=0.0, fontsize=21)
    #fig.suptitle('KPIs at district-level', fontsize=16)
    plt.tight_layout()

    fig.savefig("kpis.pdf")

    return fig


if __name__ == '__main__':
    kpis = pd.read_csv('../experiments/SAC_DB2/30_renewable_prod/reward_05pvprice/0.5/kpis_20231002T130931.csv')
    plot_district_kpis(kpis)

    plt.show()