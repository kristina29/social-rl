import time
from datetime import datetime

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from citylearn.agents.sac import SAC
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet

from scipy.stats import norm

def exploration():
    # load data
    schema = DataSet.get_schema('test')

    electricity_consumption = []
    electricity_consumption_without_storage = []
    electricity_consumption_without_storage_and_pv = []

    n = 3
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

def calculation():
    filenames = ['../experiments/Explore_Demand/01/electricity_consumption20230620T143319.csv',
                 '../experiments/Explore_Demand/02/electricity_consumption20230620T140421.csv',
                 '../experiments/Explore_Demand/3/electricity_consumption20230620T141523.csv']

    data = pd.read_csv(filenames[0])
    for file in filenames[1:]:
        data.append(pd.read_csv(file))

    x_axis = np.arange(-50, 50, 0.001)
    for col in data.columns:
        mean = data[col].mean()
        std = data[col].std()
        print(f'Mean of {col}: {mean} kWh')
        print(f'Median of {col}: {data[col].median()} kWh')
        print(f'Std of {col}: {std} kWh')
        print('')
        plt.plot(x_axis, norm.pdf(x_axis, mean, std), label=col)
    plt.legend()

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
