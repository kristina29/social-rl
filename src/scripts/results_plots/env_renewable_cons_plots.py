import pickle

import numpy as np
from matplotlib import pyplot as plt


def running_mean(x, N):
    cumsum = np.nancumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def get_possible_consumption(building, excluded_used_pv: bool) -> np.ndarray:
    ec = np.array(building['capacity_history'])

    # insert 0 in the beginning of soc such that electricity consumption of battery is shifted one time step ahead
    es = building['soc'][:-1]
    es.insert(0, 0.)
    es = np.array(es)
    max_battery_input = (ec - es * (1 - building['loss_coeff'])) / building['efficiency_history']
    battery_input = np.minimum(np.clip(max_battery_input, 0., None),
                               np.array(building['nominal_power']))

    if excluded_used_pv:
        return battery_input + building['net_electricity_consumption_without_storage_and_pv']
    else:
        return np.maximum(battery_input + building['net_electricity_consumption_without_storage_and_pv'] -
                          building['solar_generation'] * -1, 0.)


def plot_used_pv_share(data):
    figs = []

    for env_name, env_data in data.items():
        fig, ax = plt.subplots()
        for building, building_data in env_data['building_data'].items():
            could_used = get_possible_consumption(building_data, excluded_used_pv=True)

            could_have_used = np.minimum(building_data['solar_generation'] * -1, could_used)
            no_generation = np.where(building_data['solar_generation'] == 0)[0]
            could_have_used[no_generation] = 1.  # prevent errors

            used = building_data['used_pv_electricity']

            share = used / could_have_used
            share[no_generation] = 1
            share[could_have_used < 0.0001] = 1

            try:
                assert np.all(((share > -0.01) & (share < 1.01)) | np.isnan(share))
            except AssertionError:
                print('Assertion problem values:',
                      share[np.where((share <= -0.01) | (share >= 1.01))])

            y = running_mean(share, 200)
            x = range(len(y))
            ax.plot(x, y, label=building)
            ax.set_ylim(0, 1)


if __name__ == '__main__':
    file = '../experiments/SAC_DB2/all_data_20231109T155937.pkl'

    with open(file, 'rb') as file:
        envs = pickle.load(file)

    plot_used_pv_share(envs)

    plt.show()