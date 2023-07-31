import glob
from argparse import ArgumentParser

import pandas as pd
import matplotlib.pyplot as plt

EXPERIMENT_BASE_DIR = '../experiments/'

parser = ArgumentParser()
parser.add_argument('--agentdir', help='Name of the directory of chosen agent', default='SAC_DB2')
parser.add_argument('--agent', help='Name of the agent to compare KPIs of', default='SAC')
parser.add_argument('--dirs', help='Names of the directories of the experiments to compare', nargs='+')
opts = parser.parse_args()

if __name__ == '__main__':

    if False:
        experiment_dirs = parser.dirs
        agentdir = parser.agentdir
        agent = parser.agent

    if True:
        experiment_dirs = ['16_weather_8locs/standard_buildings',
                           '21_adapt_action_space/cooling_electrical/old_buildings',
                           '21_adapt_action_space/dwh_electrical/old_buildings',
                           '21_adapt_action_space/heating_electrical/old_buildings',
                           '21_adapt_action_space/all_actions/old_buildings',
                           '16_weather_8locs/new_buildings2',
                           '21_adapt_action_space/cooling_electrical/new_buildings2',
                           '21_adapt_action_space/dwh_electrical/new_buildings2',
                           '21_adapt_action_space/heating_electrical/new_buildings2',
                           '21_adapt_action_space/all_actions/new_buildings2']
        ref_dirs = ['16_weather_8locs/standard_buildings',
                    '16_weather_8locs/new_buildings2']
        agentdir = 'SAC_DB2'
        agent = 'SAC'

    kpis = {}
    for dir in experiment_dirs:
        print(dir)
        kpi_filenames = glob.glob(f'{EXPERIMENT_BASE_DIR}{agentdir}/{dir}/kpis_*.csv')

        if len(kpi_filenames) > 1:
            raise ValueError(f'More than one KPI csv file found in {dir}')

        df = pd.read_csv(kpi_filenames[0])
        df = df.set_index('kpi')
        df = df[(df['env_id'] == agent) & (df['level'] == 'district')]
        kpis[dir] = df


    # plot 1-avarage_daily_renewable share net_value/net_value_without_storage
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
    ax.axhline(values[names.index(ref_dirs[0])], xmin=0, xmax=0.5, c='red', linewidth=0.7, linestyle="-")
    ax.axhline(values[names.index(ref_dirs[1])], xmin=0.5, xmax=1, c='red', linewidth=0.7, linestyle="-")
    ax.bar_label(rects, padding=3)
    ax.set_ylim(0.95,1.05)
    ax.set_title('1 - average_daily_renewable_share [net_value/net_value_without_storage]')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    plt.savefig('1.png', bbox_inches="tight")


    # plot 1-avarage_daily_renewable share net_value/net_value_without_storage
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
    ax.axhline(values[names.index(ref_dirs[0])], xmin=0, xmax=0.5, c='red', linewidth=0.7, linestyle="-")
    ax.axhline(values[names.index(ref_dirs[1])], xmin=0.5, xmax=1, c='red', linewidth=0.7, linestyle="-")
    ax.bar_label(rects, padding=3)
    ax.set_ylim(0.95, 1.05)
    ax.set_title('1 - average_daily_renewable_share_grid [net_value/net_value_without_storage]')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    plt.savefig('2.png', bbox_inches="tight")

    # plot 1-used_pv_of_total_share net_value
    values = []
    names = []
    for key, value in kpis.items():
        values.append(round(value.loc['1 - used_pv_of_total_share', 'net_value'], 3))
        names.append(key)

    fig, ax = plt.subplots()
    ax.grid(which='major', axis='y', linestyle='--')
    ax.set_axisbelow(True)
    rects = ax.bar(names, values, label=names)
    ax.axhline(values[names.index(ref_dirs[0])], xmin=0, xmax=0.5, c='red', linewidth=0.7, linestyle="-")
    ax.axhline(values[names.index(ref_dirs[1])], xmin=0.5, xmax=1, c='red', linewidth=0.7, linestyle="-")
    ax.bar_label(rects, padding=3)
    ax.set_ylim(0, 0.2)
    ax.set_title('1 - used_pv_of_total_share [net_value]')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    plt.savefig('3.png', bbox_inches="tight")

    # plot 1-used_pv_of_total_share share net_value/net_value_without_storage
    values = []
    names = []
    for key, value in kpis.items():
        values.append(round(value.loc['1 - used_pv_of_total_share', 'net_value'] /
                            value.loc['1 - used_pv_of_total_share', 'net_value_without_storage'], 3))
        names.append(key)

    fig, ax = plt.subplots()
    ax.grid(which='major', axis='y', linestyle='--')
    ax.set_axisbelow(True)
    rects = ax.bar(names, values, label=names)
    ax.axhline(values[names.index(ref_dirs[0])], xmin=0, xmax=0.5, c='red', linewidth=0.7, linestyle="-")
    ax.axhline(values[names.index(ref_dirs[1])], xmin=0.5, xmax=1, c='red', linewidth=0.7, linestyle="-")
    ax.bar_label(rects, padding=3)
    ax.set_ylim(0.75, 1.05)
    ax.set_title('1 - used_pv_of_total_share [net_value/net_value_without_storage]')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    plt.savefig('4.png', bbox_inches="tight")






