import glob

import pandas as pd

if __name__ == '__main__':
    dirs = ['../experiments/SAC_DB2/30_renewable_prod/reward_05pvprice/0.5',
            '../experiments/SAC_DB2/30_renewable_prod/reward_05pvprice/0.5/repeats']

    kpis = pd.read_csv(glob.glob(f'{dirs[0]}/kpis_*.csv')[0])

    for file in glob.glob(f'{dirs[1]}/kpis_*.csv'):
        new_kpis = pd.read_csv(file)
        kpis = pd.concat([kpis, new_kpis])

    kpis_mean = kpis.groupby(['kpi', 'name', 'level', 'env_id']).mean().reset_index()
    kpis_std = kpis.groupby(['kpi', 'name', 'level', 'env_id']).std().reset_index()

    kpis_mean.to_csv(f'{dirs[0]}/kpis_mean.csv', index=False)
    kpis_std.to_csv(f'{dirs[0]}/kpis_std.csv', index=False)
