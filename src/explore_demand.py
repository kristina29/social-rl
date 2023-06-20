import time
from datetime import datetime

import pandas as pd

from citylearn.agents.sac import SAC
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet


def exploration():
    # load data
    schema = DataSet.get_schema('test')

    electricity_consumption = []
    electricity_consumption_without_storage = []
    electricity_consumption_without_storage_and_pv = []

    n = 10
    for i in range(n):
        # Train soft actor-critic (SAC) agent
        env = CityLearnEnv(schema)
        sac_model = SAC(env=env)
        sac_model.learn(episodes=2, deterministic_finish=True)
        electricity_consumption.extend(env.net_electricity_consumption)
        electricity_consumption_without_storage.extend(env.net_electricity_consumption_without_storage)
        electricity_consumption_without_storage_and_pv.extend(env.net_electricity_consumption_without_storage_and_pv)

    data = pd.DataFrame({'electricity_consumption': electricity_consumption,
                         'electricity_consumption_without_storage': electricity_consumption_without_storage,
                         'electricity_consumption_without_storage_and_pv': electricity_consumption_without_storage_and_pv})
    filename = f'electricity_consumption{datetime.now().strftime("%Y%m%dT%H%M%S")}.csv'
    data.to_csv(filename, index=False)
    print(f'Data stored under {filename}')
    print(f'Number of buildings: {len(env.buildings)}')
    print(f'Number of iterations: {n}')


if __name__ == '__main__':
    st = time.time()

    exploration()

    # get the end time
    et = time.time()

    # print the execution time
    elapsed_time = et - st
    print("")
    print('Execution time:', round(elapsed_time / 60), 'minutes')