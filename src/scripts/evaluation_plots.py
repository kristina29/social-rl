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
        experiment_dirs = ['23_autotune/new_buildings2/old',
                           '15_reward_price_pv/alpha075/new_buildings2/old',
                           '15_reward_price_pv/alpha05/new_buildings2/repeats',
                           '15_reward_price_pv/alpha025/new_buildings2/old',
                           '15_reward_price_pv/alpha0/new_buildings2/old',
                           ]
        ref_dirs = ['23_autotune/new_buildings2',]
        base_agent = 'SAC'
        n_refs = len(ref_dirs)
        length = 1/n_refs
        agentdir = 'SAC_DB2'#_DB2Value'
        agents = ['SAC']#, #'SAC Best']

    kpis = {}
    kpi_filenames = glob.glob(f'{EXPERIMENT_BASE_DIR}{agentdir}/{experiment_dirs[0]}/kpis_*.csv')

    if len(kpi_filenames) > 1:
        raise ValueError(f'More than one KPI csv file found in {experiment_dirs[0]}')

    df = pd.read_csv(kpi_filenames[0])
    df = df.set_index('kpi')
    df = df[(df['env_id'] == base_agent) & (df['level'] == 'district')]
    kpis[f'{ref_dirs[0]} {base_agent}'] = df

    #df = pd.read_csv(kpi_filenames[0])
    #df = df.set_index('kpi')
    #df = df[(df['env_id'] == agents[1]) & (df['level'] == 'district')]
    #kpis[f'{ref_dirs[0]} {agents[1]}'] = df


    for dir in experiment_dirs[1:]:
        print(dir)
        kpi_filenames = glob.glob(f'{EXPERIMENT_BASE_DIR}{agentdir}/{dir}/kpis_*.csv')

        if len(kpi_filenames) > 1:
            raise ValueError(f'More than one KPI csv file found in {dir}')

        for a in agents:
            df = pd.read_csv(kpi_filenames[0])
            df = df.set_index('kpi')
            df = df[(df['env_id'] == a) & (df['level'] == 'district')]
            kpis[f'{dir} {a}'] = df

    # plot 1-avarage_daily_renewable share netue/netue_without_storage
    values = []
    names = []
    for key, value in kpis.items():
        try:
            values.append(round(value.loc['1 - average_daily_renewable_share', 'net_value']/
                          value.loc['1 - average_daily_renewable_share', 'net_value_without_storage'], 3))
            names.append(key)
        except:
            pass

    fig, ax = plt.subplots()
    ax.grid(which='major', axis='y', linestyle='--')
    ax.set_axisbelow(True)
    rects = ax.bar(names, values, label=names)

    for i in range(n_refs):
        ax.axhline(values[names.index(f'{ref_dirs[i]} {base_agent}')], xmin=0+i*length, xmax=1-length*(n_refs-i-1), c='red',
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
        ax.axhline(values[names.index(f'{ref_dirs[i]} {base_agent}')], xmin=0 + i * length, xmax=1 - length * (n_refs - i - 1), c='red',
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
            ax.axhline(values[names.index(f'{ref_dirs[i]} {base_agent}')], xmin=0 + i * length, xmax=1 - length * (n_refs - i - 1),
                       c='red', linewidth=0.7, linestyle="-")
        except:
            pass
    ax.bar_label(rects, padding=3)
    ax.set_ylim(0.7, 1.05)
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
                ax.axhline(values[names.index(f'{ref_dirs[i]} {base_agent}')], xmin=0 + i * length, xmax=1 - length * (n_refs - i - 1),
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






