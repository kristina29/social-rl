import math
import os
import pickle
from typing import Tuple, List, Mapping, Iterable
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from citylearn.citylearn import CityLearnEnv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from citylearn.utilities import get_active_parts


def set_schema_buildings(schema: dict, count: int, seed: int) -> Tuple[dict, List[str]]:
    """Randomly select number of buildings to set as active in the schema.

    Parameters
    ----------
    schema: dict
        CityLearn dataset mapping used to construct environment.
    count: int
        Number of buildings to set as active in schema.
    seed: int
        Seed for pseudo-random number generator

    Returns
    -------
    schema: dict
        CityLearn dataset mapping with active buildings set.
    buildings: List[str]
        List of selected buildings.
    """

    assert 1 <= count <= 15, 'Count must be between 1 and 15.'

    # set random seed
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(27) #always get the same buildings for training

    # get all building names
    buildings = list(schema['buildings'].keys())

    # remove buildins 12 and 15 as they have pecularities in their data
    # that are not relevant to this tutorial
    buildings_to_exclude = ['Building_12', 'Building_15']

    for b in buildings_to_exclude:
        if b in buildings:
            buildings.remove(b)

    # randomly select specified number of buildings
    buildings = np.random.choice(buildings, size=count, replace=False).tolist()

    # reorder buildings
    building_ids = [int(b.split('_')[-1]) for b in buildings]
    building_ids = sorted(building_ids)
    buildings = [f'Building_{i}' for i in building_ids]

    # update schema to only included selected buildings
    for b in schema['buildings']:
        if b in buildings:
            schema['buildings'][b]['include'] = True
        else:
            schema['buildings'][b]['include'] = False

    np.random.seed()

    return schema, buildings


def set_schema_demonstrators(schema: dict, count: int, seed: int) -> Tuple[dict, List[str]]:
    """Randomly select number of buildings to use as demonstrators.

    Parameters
    ----------
    schema: dict
        CityLearn dataset mapping used to construct environment.
    count: int
        Number of buildings to set as demonstrator in schema.
    seed: int
        Seed for pseudo-random number generator

    Returns
    -------
    schema: dict
        CityLearn dataset mapping with active buildings set.
    demonstrators: List[str]
        List of selected buildings.
    """

    # set random seed
    if seed is not None:
        np.random.seed(seed)

    # get all active buildings
    buildings = get_active_parts(schema, key='buildings')

    # randomly select specified number of buildings
    demonstrators = np.random.choice(buildings, size=count, replace=False).tolist()

    # update schema to only included selected buildings
    for b in schema['buildings']:
        if b in demonstrators:
            schema['buildings'][b]['demonstrator'] = True
        else:
            schema['buildings'][b]['demonstrator'] = False

    return schema, demonstrators


def set_schema_simulation_period(schema: dict, count: int, seed: int, root_directory=None) -> Tuple[dict, int, int]:
    """Randomly select environment simulation start and end time steps
    that cover a specified number of days.

    Parameters
    ----------
    schema: dict
        CityLearn dataset mapping used to construct environment.
    count: int
        Number of simulation days.
    seed: int
        Seed for pseudo-random number generator.
    root_directory:

    Returns
    -------
    schema: dict
        CityLearn dataset mapping with `simulation_start_time_step`
        and `simulation_end_time_step` key-values set.
    simulation_start_time_step: int
        The first time step in schema time series files to
        be read when constructing the environment.
    simulation_end_time_step: int
        The last time step in schema time series files to
        be read when constructing the environment.
    """

    assert 1 <= count <= 365, 'Count must be between 1 and 365.'

    # set random seed
    np.random.seed(seed)

    # use any of the files to determine the total
    # number of available time steps
    filename = schema['buildings']['Building_1']['carbon_intensity']
    filepath = os.path.join(root_directory, filename)
    time_steps = pd.read_csv(filepath).shape[0]

    # set candidate simulation start time steps
    # spaced by the number of specified days
    simulation_start_time_step_list = np.arange(0, time_steps, 24 * count)

    # randomly select a simulation start time step
    simulation_start_time_step = np.random.choice(
        simulation_start_time_step_list, size=1
    )[0]
    simulation_end_time_step = simulation_start_time_step + 24 * count - 1

    # update schema simulation time steps
    schema['simulation_start_time_step'] = simulation_start_time_step
    schema['simulation_end_time_step'] = simulation_end_time_step

    return schema, simulation_start_time_step, simulation_end_time_step


def set_active_observations(schema: dict, active_observations: List[str]) -> Tuple[dict, List[str]]:
    """Set the observations that will be part of the environment's
    observation space that is provided to the control agent.

    Parameters
    ----------
    schema: dict
        CityLearn dataset mapping used to construct environment.
    active_observations: List[str]
        Names of observations to set active to be passed to control agent.

    Returns
    -------
    schema: dict
        CityLearn dataset mapping with active observations set.
    observations: List[str]
        List of active observations.
    """

    active_count = 0
    valid_observations = list(schema['observations'].keys())
    observations = []

    for o in schema['observations']:
        if o in active_observations:
            schema['observations'][o]['active'] = True
            observations.append(o)
            active_count += 1
        else:
            schema['observations'][o]['active'] = False

    assert active_count > 0, \
        'the provided observations are not valid observations.' \
        f' Valid observations in CityLearn are: {valid_observations}'

    return schema, observations


def get_kpis(env: CityLearnEnv) -> pd.DataFrame:
    """Returns evaluation KPIs.

    Electricity consumption, cost and carbon emissions KPIs are provided
    at the building-level and average district-level. Average daily peak,
    ramping and (1 - load factor) KPIs are provided at the district level.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment instance.

    Returns
    -------
    kpis: pd.DataFrame
        KPI table.
    """

    kpis = env.evaluate()

    # names of KPIs to retrieve from evaluate function
    kpi_names = [
        'electricity_consumption', 'cost',
        'average_daily_peak', 'ramping', '1 - load_factor',
        '1 - average_daily_renewable_share',
        '1 - average_daily_renewable_share_grid',
        '1 - used_pv_of_total_share'
    ]
    kpis = kpis[
        (kpis['cost_function'].isin(kpi_names))
    ].dropna()

    # round up the values to 3 decimal places for readability
    kpis['value'] = kpis['value'].round(3)
    kpis['net_value'] = kpis['net_value'].round(3)
    kpis['net_value_without_storage'] = kpis['net_value_without_storage'].round(3)

    # rename the column that defines the KPIs
    kpis = kpis.rename(columns={'cost_function': 'kpi'})

    return kpis


def plot_building_kpis(envs: Mapping[str, CityLearnEnv]) -> plt.Figure:
    """Plots electricity consumption, cost and carbon emissions
    at the building-level for different control agents in bar charts.

    Parameters
    ----------
    envs: Mapping[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """

    kpis_list = []

    for k, v in envs.items():
        kpis = get_kpis(v)
        kpis = kpis[kpis['level'] == 'building'].copy()
        kpis['building_id'] = kpis['name'].str.split('_', expand=True)[1]
        kpis['building_id'] = kpis['building_id'].astype(int).astype(str)
        kpis['env_id'] = k
        kpis = kpis.drop(columns=['net_value', 'net_value_without_storage'])
        kpis_list.append(kpis)

    kpis = pd.concat(kpis_list, ignore_index=True, sort=False)
    kpi_names = kpis['kpi'].unique()
    column_count_limit = 8
    row_count = math.ceil(len(kpi_names) / column_count_limit)
    column_count = min(column_count_limit, len(kpi_names))
    building_count = len(kpis['name'].unique())
    env_count = len(envs)
    figsize = (3.0 * column_count, 0.3 * env_count * building_count * row_count)
    fig, _ = plt.subplots(
        row_count, column_count, figsize=figsize, sharey=True
    )

    for i, (ax, (k, k_data)) in enumerate(zip(fig.axes, kpis.groupby('kpi'))):
        sns.barplot(x='value', y='name', data=k_data, hue='env_id', ax=ax)
        ax.axvline(1.0, color='black', linestyle='--', label='Baseline')
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_title(k)

        if i == len(kpi_names) - 1:
            ax.legend(
                loc='upper left', bbox_to_anchor=(1.3, 1.0), framealpha=0.0
            )
        else:
            ax.legend().set_visible(False)

        for s in ['right', 'top']:
            ax.spines[s].set_visible(False)

        for p in ax.patches:
            ax.text(
                p.get_x() + p.get_width(),
                p.get_y() + p.get_height() / 2.0,
                p.get_width(), ha='left', va='center'
            )

    fig.suptitle('KPIs at building-level', fontsize=16)
    plt.tight_layout()
    return fig


def plot_district_kpis(envs: Mapping[str, CityLearnEnv]) -> plt.Figure:
    """Plots electricity consumption, cost, carbon emissions,
    average daily peak, ramping and (1 - load factor) at the
    district-level for different control agents in a bar chart.

    Parameters
    ----------
    envs: Mapping[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """

    kpis_list = []

    for k, v in envs.items():
        kpis = get_kpis(v)
        kpis = kpis[kpis['level'] == 'district'].copy()
        kpis['env_id'] = k
        kpis = kpis.drop(columns=['net_value', 'net_value_without_storage'])
        kpis_list.append(kpis)

    kpis = pd.concat(kpis_list, ignore_index=True, sort=False)
    row_count = 1
    column_count = 1
    env_count = len(envs)
    kpi_count = len(kpis['kpi'].unique())
    figsize = (6.0 * column_count, 0.225 * env_count * kpi_count * row_count)
    fig, ax = plt.subplots(row_count, column_count, figsize=figsize)
    sns.barplot(x='value', y='kpi', data=kpis, hue='env_id', ax=ax)
    ax.axvline(1.0, color='black', linestyle='--', label='Baseline')
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    for s in ['right', 'top']:
        ax.spines[s].set_visible(False)

    for p in ax.patches:
        ax.text(
            p.get_x() + p.get_width(),
            p.get_y() + p.get_height() / 2.0,
            p.get_width(), ha='left', va='center'
        )

    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0), framealpha=0.0)
    fig.suptitle('KPIs at district-level', fontsize=16)
    plt.tight_layout()

    return fig


def plot_building_load_profiles(envs: Mapping[str, CityLearnEnv]) -> plt.Figure:
    """Plots building-level net electricty consumption profile
    for different control agents.

    Parameters
    ----------
    envs: Mapping[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """

    building_count = len(list(envs.values())[0].buildings)
    column_count_limit = 8
    row_count = math.ceil(building_count / column_count_limit)
    column_count = min(column_count_limit, building_count)
    figsize = (4.0 * column_count, 1.75 * row_count)
    fig, _ = plt.subplots(row_count, column_count, figsize=figsize)

    for i, ax in enumerate(fig.axes):
        for k, v in envs.items():
            y = v.buildings[i].net_electricity_consumption[:168]
            x = range(len(y))
            ax.plot(x, y, label=k)

        y = v.buildings[i].net_electricity_consumption_without_storage[:168]
        ax.plot(x, y, label='Baseline')
        ax.set_title(v.buildings[i].name)
        ax.set_xlabel('Time')
        ax.set_ylabel('kWh')
        ticks = get_weekday_ticks()
        ax.set_xticks(np.arange(0, len(ticks)))
        ax.xaxis.set_tick_params(length=0)
        ax.set_xticklabels(ticks, rotation=0)
        [l.set_visible(False) for (i, l) in enumerate(ax.get_xticklabels()) if (i - 18) % 24 != 0]

        if i == building_count - 1:
            ax.legend(
                loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=0.0
            )
        else:
            ax.legend().set_visible(False)

    fig.suptitle('Building-level net electricty consumption profile', fontsize=16)
    plt.tight_layout()

    return fig


def get_weekday_ticks():
    ticks = ['Mo'] * 24
    ticks.extend(['Tue'] * 24)
    ticks.extend(['Wed'] * 24)
    ticks.extend(['Thur'] * 24)
    ticks.extend(['Fri'] * 24)
    ticks.extend(['Sat'] * 24)
    ticks.extend(['Sun'] * 24)
    return ticks


def plot_district_load_profiles(envs: Mapping[str, CityLearnEnv]) -> plt.Figure:
    """Plots district-level net electricty consumption profile
    for different control agents.

    Parameters
    ----------
    envs: Mapping[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """

    figsize = (5.0, 1.5)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for k, v in envs.items():
        y = v.net_electricity_consumption[:168]
        x = range(len(y))
        ax.plot(x, y, label=k)

    y = v.net_electricity_consumption_without_storage[:168]
    ax.plot(x, y, label='Baseline')
    ax.set_xlabel('Time')
    ax.set_ylabel('kWh')
    ticks = get_weekday_ticks()
    ax.set_xticks(np.arange(0, len(ticks)))
    ax.xaxis.set_tick_params(length=0)
    ax.set_xticklabels(ticks, rotation=0)
    [l.set_visible(False) for (i, l) in enumerate(ax.get_xticklabels()) if (i - 18) % 24 != 0]
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=0.0)

    fig.suptitle('District-level net electricty consumption profile', fontsize=14)
    plt.tight_layout()
    return fig


def plot_renewable_share(envs: Mapping[str, CityLearnEnv], grid: bool=False) -> plt.Figure:
    """Plots renewable share KPIs over time for different control agents.

    Parameters
    ----------
    envs: Mapping[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.
    grid: bool
        Indicates if renewable share only from grid or total (including building PVs) should be plottet

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """

    #figsize = (5.0, 1.5)
    fig, ax = plt.subplots(1, 1)#, figsize=figsize)

    for k, v in envs.items():
        if grid:
            could_used = 0
            for b in v.buildings:
                e = np.array(b.net_electricity_consumption)
                ec = np.array(b.electrical_storage.capacity_history)
                es = np.array(b.electrical_storage.soc)
                could_used += (ec - es) + np.clip(e, 0., None) - np.clip(b.electrical_storage_electricity_consumption,
                                                                         0., None)
            could_have_used = (v.buildings[0].fuel_mix.renewable_energy_produced / could_used).clip(None, 1.)
            used = (v.net_renewable_electricity_grid_consumption / could_used).clip(None, 1.)
            y = running_mean(used / could_have_used, 160)
        else:
            could_used = 0
            for b in v.buildings:
                e = np.array(b.net_electricity_consumption)
                ec = np.array(b.electrical_storage.capacity_history)
                es = np.array(b.electrical_storage.soc)
                could_used += (ec-es) + np.clip(e, 0., None) - np.clip(b.electrical_storage_electricity_consumption, 0., None)
            could_have_used = ((v.buildings[0].fuel_mix.renewable_energy_produced+v.solar_generation*-1)/could_used).clip(None, 1.)
            used = (v.net_renewable_electricity_consumption/could_used).clip(None, 1.)
            y = running_mean(used/could_have_used, 160)
        x = range(len(y))
        ax.plot(x, y, label=k)
        ax.set_ylim(0, 1)

    if grid:
        y = running_mean(
            v.net_renewable_electricity_grid_share_without_storage/v.buildings[0].fuel_mix.renewable_energy_share, 160)
    else:
        y = running_mean(
            v.net_renewable_electricity_share_without_storage/v.buildings[0].fuel_mix.renewable_energy_share, 160)
    # ax.plot(x, y, label='Baseline')

    # available_renewable_share = running_mean(v.buildings[0].fuel_mix.renewable_energy_share, 160)
    # ax.plot(x, available_renewable_share, label='Available renewable share')

    ax.set_xlabel('Time')
    ax.set_ylabel('%')
    ax.xaxis.set_tick_params(length=0)
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=0.0)

    if grid:
        title = 'District-level renewable energy share grid /  available share'
    else:
        title = 'District-level renewable energy share /  available share'
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_battery_soc_profiles(envs: Mapping[str, CityLearnEnv]) -> plt.Figure:
    """Plots building-level battery SoC profiles from different control agents.

    Parameters
    ----------
    envs: Mapping[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """

    building_count = len(list(envs.values())[0].buildings)
    column_count_limit = 8
    row_count = math.ceil(building_count / column_count_limit)
    column_count = min(column_count_limit, building_count)
    figsize = (4.0 * column_count, 1.75 * row_count)
    fig, _ = plt.subplots(row_count, column_count, figsize=figsize)

    for i, ax in enumerate(fig.axes):
        for k, v in envs.items():
            soc = np.array(v.buildings[i].electrical_storage.soc)[:168]
            capacity = v.buildings[i].electrical_storage.capacity_history[0]
            y = soc / capacity
            x = range(len(y))
            ax.plot(x, y, label=k)

        ax.set_title(v.buildings[i].name)
        ax.set_xlabel('Time')
        ax.set_ylabel('SoC')
        ticks = get_weekday_ticks()
        ax.set_xticks(np.arange(0, len(ticks)))
        ax.xaxis.set_tick_params(length=0)
        ax.set_xticklabels(ticks, rotation=0)
        [l.set_visible(False) for (i, l) in enumerate(ax.get_xticklabels()) if (i - 18) % 24 != 0]

        if i == building_count - 1:
            ax.legend(
                loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=0.0
            )
        else:
            ax.legend().set_visible(False)

    fig.suptitle('Building-level battery SoC profiles', fontsize=16)
    plt.tight_layout()

    return fig


def plot_rewards(rewards: Mapping[str, List[List[float]]], envs: Mapping[str, CityLearnEnv]) -> List[plt.Figure]:
    r"""Creates one figure over time of the rewards for each agent.

        Parameters
        ----------
        rewards : Mapping[str, List[float]]
            Reward values for each building of each trained agent over time.
        envs: Mapping[str, CityLearnEnv]
            Mapping of user-defined control agent names to environments
            the agents have been used to control.

        Return values
        ----------
        List of the created figures
    """
    figs = []
    for env_name, env_rewards in rewards.items():
        buldings = envs[env_name].buildings
        for building_index, rewards in enumerate(env_rewards):
            fig, ax = plt.subplots()

            N = int(len(rewards) / 50)
            ax.plot(np.arange(len(rewards) - N + 1) * N, running_mean(rewards, N))
            ax.set_ylabel(f'Reward value (running mean over {N} values)')
            ax.set_xlabel(f'Time step (actual {len(rewards)})')

            ax.set_title(f'{env_name} - {buldings[building_index].name} reward values')
            ax.grid(axis='y')
            figs.append(fig)

    return figs


def plot_losses(losses: Mapping[str, Mapping[int, Mapping[str, List[float]]]],
                envs: Mapping[str, CityLearnEnv]) -> List[plt.Figure]:
    r"""Creates one figure over time of the losses for each building for each agent.

        Parameters
        ----------
        losses : Mapping[str, Mapping[int, Mapping[str, List[float]]]]
            Loss values for each trained agent over time.
                Mapping from agent name to all losses of this agent.
                Mapping from building index to all losses of this building.
                Mapping from neural network name to losses of this network.
        envs: Mapping[str, CityLearnEnv]
            Mapping of user-defined control agent names to environments
            the agents have been used to control.

        Return values
        ----------
        List of the created figures
    """

    figs = []
    for env_name, env_values in losses.items():
        buldings = envs[env_name].buildings
        for building_index, building_losses in env_values.items():
            fig, ax = plt.subplots()
            for nn_name, nn_values in building_losses.items():
                ax.plot(np.arange(len(nn_values)), nn_values, label=nn_name)
            ax.legend()
            ax.set_ylabel(f'Loss value')
            ax.set_xlabel('Time step')
            ax.set_title(f'{env_name} - {buldings[building_index].name} loss values')
            ax.grid(axis='y')
            figs.append(fig)

    return figs


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# save all produces figures as pdf
def save_multi_image(filename):
    """https://www.tutorialspoint.com/saving-multiple-figures-to-one-pdf-file-in-matplotlib"""
    pp = PdfPages(filename + '.pdf')
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
        plt.close(fig)
    pp.close()
    print("Plots saved in file", f'{filename}.pdf')


def plot_simulation_summary(envs: Mapping[str, CityLearnEnv], losses: Mapping[str, Mapping[str, List[float]]],
                            rewards: Mapping[str, List[List[float]]], filename: str):
    """Plots KPIs, load and battery SoC profiles for different control agents.

    Parameters
    -----c-----
    envs: Mapping[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.
    losses: Mapping[str, Mapping[str, List[float]]]
        Mapping of user-defined control agent names to Mapping of neural-network name to loss values of training steps.
    rewards: Mapping[str, List[float]]
        Mapping of user-defined control agent names to rewards of training steps.
    filename: str
        Name of the file where plots should be stored
    """

    plot_building_kpis(envs)
    plot_building_load_profiles(envs)
    plot_battery_soc_profiles(envs)
    plot_district_kpis(envs)
    plot_district_load_profiles(envs)
    plot_renewable_share(envs)
    plot_renewable_share(envs, grid=True)
    plot_losses(losses, envs)
    plot_rewards(rewards, envs)

    save_multi_image(filename)


def save_kpis(envs: Mapping[str, CityLearnEnv], filename):
    """Saves KPIs for different control agents as CSV.

    Parameters
    -----c-----
    envs: Mapping[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.
    filename: str
        Name of CSV file where KPIs should be stored
    """

    kpis_list = []

    for k, v in envs.items():
        kpis = get_kpis(v)
        kpis['env_id'] = k
        kpis_list.append(kpis)

    kpis = pd.concat(kpis_list, ignore_index=True, sort=False)
    kpis.to_csv(filename, index=False)


def save_results(envs: Mapping[str, CityLearnEnv], losses: Mapping[str, Mapping[str, List[float]]],
                 rewards: Mapping[str, List[List[float]]]):

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    # plot summary and compare with other control results
    p_filename = f'plots_{timestamp}'
    plot_simulation_summary(envs, losses, rewards, p_filename)

    # save KPIs as csv
    k_filename = f'kpis_{timestamp}.csv'
    save_kpis(envs, k_filename)
    print(f'KPIs saved to {k_filename}')

    # save losses to losses.pkl file
    l_filename = f'losses_{timestamp}.pkl'
    with open(l_filename, 'wb') as fp:
        pickle.dump(losses, fp)
        print(f'Losses dictionary saved to {l_filename}')

    # save rewards to rewards.pkl file
    r_filename = f'rewards_{timestamp}.pkl'
    with open(r_filename, 'wb') as fp:
        pickle.dump(rewards, fp)
        print(f'Rewards dictionary saved to {r_filename}')

    print('')
    print('---------------------------------')
    print('COPY COMMANDS')
    print('---------------------------------')
    print(f'scp klietz10@134.2.168.52:/mnt/qb/work/ludwig/klietz10/social-rl/{p_filename}.pdf '
          f'experiments/SAC_DB2/{p_filename}.pdf')
    print(f'scp klietz10@134.2.168.52:/mnt/qb/work/ludwig/klietz10/social-rl/{k_filename} '
          f'experiments/SAC_DB2/{k_filename}')
    print(f'scp klietz10@134.2.168.52:/mnt/qb/work/ludwig/klietz10/social-rl/{l_filename} '
          f'experiments/SAC_DB2/{l_filename}')
    print(f'scp klietz10@134.2.168.52:/mnt/qb/work/ludwig/klietz10/social-rl/{r_filename} '
          f'experiments/SAC_DB2/{r_filename}')