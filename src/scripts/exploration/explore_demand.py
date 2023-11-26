import time
from datetime import datetime

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from citylearn.agents.sac import SAC
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet

from scipy.stats import norm

from nonsocialrl import preprocessing


def exploration():
    # load data
    schema = DataSet.get_schema('test')

    # schema = preprocessing(schema, 2, 0, ['hour'])

    electricity_consumption = []
    electricity_consumption_without_storage = []
    electricity_consumption_without_storage_and_pv = []
    building_id = []

    n = 3
    for i in range(n):
        # Train soft actor-critic (SAC) agent
        env = CityLearnEnv(schema)
        sac_model = SAC(env=env)
        sac_model.learn(episodes=2, deterministic_finish=True)
        for b in env.buildings:
            electricity_consumption.extend(b.net_electricity_consumption)
            electricity_consumption_without_storage.extend(b.net_electricity_consumption_without_storage)
            electricity_consumption_without_storage_and_pv.extend(b.net_electricity_consumption_without_storage_and_pv)
            building_id.extend([b.name] * len(b.net_electricity_consumption))

    data = pd.DataFrame({'electricity_consumption': electricity_consumption,
                         'electricity_consumption_without_storage': electricity_consumption_without_storage,
                         'electricity_consumption_without_storage_and_pv': electricity_consumption_without_storage_and_pv,
                         'building': building_id})
    filename = f'electricity_consumption{datetime.now().strftime("%Y%m%dT%H%M%S")}.csv'
    data.to_csv(filename, index=False)
    print(f'Data stored under {filename}')
    print(f'Number of buildings: {len(env.buildings)}')
    print(f'Number of iterations: {n}')


def calculation():
    filenames = ['../experiments/Explore_Demand/per_building/electricity_consumption20230621T154955.csv',
                 '../experiments/Explore_Demand/per_building/electricity_consumption20230621T155019.csv',
                 '../experiments/Explore_Demand/per_building/electricity_consumption20230621T155100.csv',
                 '../experiments/Explore_Demand/per_building/electricity_consumption20230621T160455.csv']

    data = pd.read_csv(filenames[0])
    for file in filenames[1:]:
        data.append(pd.read_csv(file))

    data = data.groupby('building', as_index=False).agg(['median', 'std'])

    stat_data = pd.DataFrame(columns=['Column', 'Median [kWh]', 'Std [kWh]', 'Building'])
    for i in range(0, len(data.columns), 2):
        col = data.columns[i]
        stat_data = stat_data.append(pd.DataFrame({'Column': col[0],
                                                   'Median [kWh]': data[col].tolist(),
                                                   'Std [kWh]': data.iloc[:, i + 1].tolist(),
                                                   'Building': data.index.tolist()}))

    stat_data.to_csv('citylearn/data/nydata/statistics_building_electricity_consumption.csv', index=False)


if __name__ == '__main__':
    st = time.time()

    explore = False
    calculate = True

    if explore:
        exploration()
    if calculate:
        calculation()

    # get the end time
    et = time.time()

    # print the execution time
    elapsed_time = et - st
    print("")
    print('Execution time:', round(elapsed_time / 60), 'minutes')
