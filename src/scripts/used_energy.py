import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_percent_greater(array1, array2):
    not_nan_mask = ~np.logical_or(np.isnan(array1), np.isnan(array2))
    # Use the mask to perform element-wise comparison only where not NaN in both arrays
    comparison = np.greater(array1[not_nan_mask], array2[not_nan_mask])
    # Count the number of True values where array1 is greater than array2
    num_greater = np.sum(comparison)
    # Calculate the percentage based on the number of non-NaN comparisons made
    percent_greater = (num_greater / not_nan_mask.sum()) * 100
    print('percent_greater', percent_greater)

    difference = array1[not_nan_mask] - array2[not_nan_mask]
    mean_greater = difference.mean()
    print('mean_greater', mean_greater)


def running_mean(x, N):
    cumsum = np.nancumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def get_possible_consumption(building_data, excluded_used_pv) -> np.ndarray:
    ec = np.array(building_data['capacity_history'])

    # insert 0 in the beginning of soc such that electricity consumption of battery is shifted one time step ahead
    es = building_data['soc'][:-1]
    es.insert(0, 0.)
    es = np.array(es)
    max_battery_input = (ec - es * (1 - building_data['loss_coeff'])) / building_data['efficiency_history']
    battery_input = np.minimum(np.clip(max_battery_input, 0., None),
                               np.array(building_data['nominal_power']))

    if excluded_used_pv:
        return battery_input + building_data['net_electricity_consumption_without_storage_and_pv']
    else:
        return np.maximum(battery_input + building_data['net_electricity_consumption_without_storage_and_pv'] -
                          building_data['solar_generation'] * -1, 0.)


def plot_grid_consumption(data):
    fig, ax = plt.subplots()

    i = 5

    ys = []

    for label, agent_data in data.items():
        could_used = 0
        for building, b_data in agent_data['building_data'].items():
            could_used += get_possible_consumption(b_data, excluded_used_pv=False)

        could_have_used = np.minimum(agent_data['renewable_energy_produced'], could_used)
        used = agent_data['net_renewable_electricity_grid_consumption']

        share = used / could_have_used
        print(share[833])
        # share[share == np.inf] = -2
        # share[could_have_used < 0.0001] = -2

        #try:
        #    assert np.all(((share > -0.01) & (share < 1.01)) | np.isnan(share))
        #except AssertionError:
        #    print('Assertion problem values:',
        #          share[np.where((share <= -0.01) | (share >= 1.01))])

        y = running_mean(share, 168)
        ys.append(y)
        #y = share
        x = range(len(y))
        ax.plot(x, y, label=label, zorder=i)
        ax.set_ylim(0, 1)

    get_percent_greater(ys[1], ys[0])

    ax.set_xlabel('Time')
    ax.set_ylabel('%')
    ax.xaxis.set_tick_params(length=0)
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=0.0)
    plt.tight_layout()
    plt.show()


def plot_pv_consumption(data):

    for building, b_data in data['SAC']['building_data'].items():
        fig, ax = plt.subplots()
        ys = []
        for label, agent_data in data.items():
            b_data = agent_data['building_data'][building]

            could_used = get_possible_consumption(b_data, excluded_used_pv=True)

            could_have_used = np.minimum(b_data['solar_generation'] * -1, could_used)
            no_generation = np.where(b_data['solar_generation'] == 0)[0]
            could_have_used[no_generation] = 1.  # prevent errors

            used = b_data['used_pv_electricity']

            share = used / could_have_used
            share[no_generation] = 1

            y = running_mean(share, 168)
            ys.append(y)
            #y = share
            x = range(len(y))
            ax.plot(x, y, label=label, zorder=i)
            ax.set_ylim(0, 1)

        print(building)
        get_percent_greater(ys[1], ys[0])
        print('')

        ax.set_xlabel('Time')
        ax.set_ylabel('%')
        ax.xaxis.set_tick_params(length=0)
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=0.0)
        fig.suptitle(f'Used PV /  available share ({building})', fontsize=14)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    files = ['../experiments/New_Buildings/SAC Baseline/all_data_20231113T132158.pkl',
             '../experiments/New_Buildings/Demo_B6/extra_pol/ir0.25/all_data_20231113T164524.pkl']
    names = ['SAC', 'SAC-DemoQ']
    agents = ['SAC Best', 'SAC_DB2Value Best']

    data = {}

    for i, file in enumerate(files):
        with open(file, 'rb') as file:
            data[names[i]] = pickle.load(file)[agents[i]]

    plot_grid_consumption(data)
    plot_pv_consumption(data)






