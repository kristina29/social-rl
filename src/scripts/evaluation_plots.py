import glob
from argparse import ArgumentParser

import pandas as pd
import matplotlib.pyplot as plt

EXPERIMENT_BASE_DIR = '../experiments/'

parser = ArgumentParser()
parser.add_argument('--agentdir', help='Name of the directory of chosen agent', default='SAC_DB2')
parser.add_argument('--agent1', help='Name of the agent to compare KPIs of', default='SAC')
parser.add_argument('--agent2', help='Name of the agent to compare KPIs of', default='SAC_DB2')
parser.add_argument('--dirs', help='Names of the directories of the experiments to compare', nargs='+')
opts = parser.parse_args()

if __name__ == '__main__':

    if False:
        experiment_dirs = parser.dirs
        agentdir = parser.agentdir
        agent = parser.agent

    if True:
        experiment_dirs = ['SAC_DB2/30_renewable_prod/reward_05pvprice/0.5',
                           '4_demo_b6_policyupdate/ir0.00001',
                           '4_demo_b6_policyupdate/ir0.0001',
                           '4_demo_b6_policyupdate/ir0.001',
                           '4_demo_b6_policyupdate/ir0.01',
                           '4_demo_b6_policyupdate/ir0.02',
                           '4_demo_b6_policyupdate/ir0.03',
                           '4_demo_b6_policyupdate/ir0.04',
                           '4_demo_b6_policyupdate/ir0.05',
                           #'1_randomdemo/d4/ir0.0001',
                           #'1_randomdemo/d4/ir0.001',
                           #'1_randomdemo/d4/ir0.01',
                           #'1_randomdemo/d4/ir0.02',
                           ]
        ref_dirs = ['30_renewable_prod/reward_05pvprice/0.5']
        asocial_agent = 'SAC'
        n_refs = len(ref_dirs)
        length = 1/n_refs
        agentdir = 'SAC_DB2Value'
        agent = 'SAC_DB2Value'

    kpis = {}
    kpi_filenames = glob.glob(f'{EXPERIMENT_BASE_DIR}/{experiment_dirs[0]}/kpis_*.csv')

    if len(kpi_filenames) > 1:
        raise ValueError(f'More than one KPI csv file found in {experiment_dirs[0]}')

    df = pd.read_csv(kpi_filenames[0])
    df = df.set_index('kpi')
    df = df[(df['env_id'] == asocial_agent) & (df['level'] == 'district')]
    kpis[ref_dirs[0]] = df

    for dir in experiment_dirs[1:]:
        print(dir)
        kpi_filenames = glob.glob(f'{EXPERIMENT_BASE_DIR}{agentdir}/{dir}/kpis_*.csv')

        if len(kpi_filenames) > 1:
            raise ValueError(f'More than one KPI csv file found in {dir}')

        df = pd.read_csv(kpi_filenames[0])
        df = df.set_index('kpi')
        df = df[(df['env_id'] == agent) & (df['level'] == 'district')]
        kpis[dir] = df

    # plot 1-avarage_daily_renewable share netue/netue_without_storage
    values = []
    names = []
    for key, value in kpis.items():
        values.append(round(value.loc['1 - average_daily_renewable_share', 'net_value']/
                      value.loc['1 - average_daily_renewable_share', 'net_value_without_storage'], 3))
        names.append(key)

    fig, ax = plt.subplots()
    ax.grid(which='major', axis='y', linestyle='--')
    ax.set_axisbelow(True)
    rects = ax.bar(names, values, label=names)

    for i in range(n_refs):
        ax.axhline(values[names.index(ref_dirs[i])], xmin=0+i*length, xmax=1-length*(n_refs-i-1), c='red',
                   linewidth=0.7, linestyle="-")
    ax.bar_label(rects, padding=3)
    ax.set_ylim(0.95,1.05)
    ax.set_title('1 - average_daily_renewable_share [net_value/net_value_without_storage]')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    plt.savefig('1.png', bbox_inches="tight")


    # plot 1-avarage_daily_renewable share netue/netue_without_storage
    values = []
    names = []
    for key, value in kpis.items():
        values.append(round(value.loc['1 - average_daily_renewable_share_grid', 'net_value'] /
                            value.loc['1 - average_daily_renewable_share_grid', 'net_value_without_storage'], 3))
        names.append(key)

    fig, ax = plt.subplots()
    ax.grid(which='major', axis='y', linestyle='--')
    ax.set_axisbelow(True)
    rects = ax.bar(names, values, label=names)
    for i in range(n_refs):
        ax.axhline(values[names.index(ref_dirs[i])], xmin=0 + i * length, xmax=1 - length * (n_refs - i - 1), c='red',
                   linewidth=0.7, linestyle="-")
    ax.bar_label(rects, padding=3)
    ax.set_ylim(0.95, 1.05)
    ax.set_title('1 - average_daily_renewable_share_grid [net_value/net_value_without_storage]')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    plt.savefig('2.png', bbox_inches="tight")

    # plot 1-used_pv_of_total_share share netue/netue_without_storage
    values = []
    names = []
    for key, value in kpis.items():
        if '1 - used_pv_of_total_share' in value.index:
            values.append(round(value.loc['1 - used_pv_of_total_share', 'net_value'] /
                                value.loc['1 - used_pv_of_total_share', 'net_value_without_storage'], 3))
            names.append(key)

    fig, ax = plt.subplots()
    ax.grid(which='major', axis='y', linestyle='--')
    ax.set_axisbelow(True)
    rects = ax.bar(names, values, label=names)
    for i in range(n_refs):
        try:
            ax.axhline(values[names.index(ref_dirs[i])], xmin=0 + i * length, xmax=1 - length * (n_refs - i - 1),
                       c='red', linewidth=0.7, linestyle="-")
        except:
            pass
    ax.bar_label(rects, padding=3)
    ax.set_ylim(0.75, 1.05)
    ax.set_title('1 - used_pv_of_total_share [net_value/net_value_without_storage]')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    plt.savefig('3.png', bbox_inches="tight")

    try:
        # plot fossil_energy_consumption
        values = []
        names = []
        for key, value in kpis.items():
            if 'fossil_energy_consumption' in value.index:
                values.append(round(value.loc['fossil_energy_consumption', 'net_value'] /
                                    value.loc['fossil_energy_consumption', 'net_value_without_storage'], 3))
                names.append(key)

        fig, ax = plt.subplots()
        ax.grid(which='major', axis='y', linestyle='--')
        ax.set_axisbelow(True)
        rects = ax.bar(names, values, label=names)
        for i in range(n_refs):
            try:
                ax.axhline(values[names.index(ref_dirs[i])], xmin=0 + i * length, xmax=1 - length * (n_refs - i - 1),
                           c='red', linewidth=0.7, linestyle="-")
            except:
                pass
        ax.bar_label(rects, padding=3)
        ax.set_ylim(0.75, 1.05)
        ax.set_title('fossil_energy_consumption [netue/netue_without_storage]')
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)

        plt.savefig('4.png', bbox_inches="tight")
    except:
        pass






