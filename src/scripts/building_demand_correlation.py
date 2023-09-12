import pandas as pd
import numpy as np

from scipy.stats import pearsonr

DIR_PATH = '../citylearn/data/nydata_new_buildings2'


def read_building_demand(b_id):
    df = pd.read_csv(f'{DIR_PATH}/Building_{b_id}.csv')
    return np.array(df['Equipment Electric Power [kWh]'])


if __name__ == '__main__':
    n_buildings = 17

    demands = {}
    for b_id in range(1, n_buildings + 1):
        demands[b_id] = read_building_demand(b_id)

    cols = list(range(1, n_buildings+1))
    cols.append('Median')
    correlations = pd.DataFrame(columns=cols,
                                index=list(range(1, n_buildings+1)))
    for b_id, demand in demands.items():
        print(b_id)
        for b_id2, demand2 in demands.items():
            val = correlations.iloc[b_id-1][b_id2]
            if np.isnan(val):
                correlations.iloc[b_id-1][b_id2] = pearsonr(demand, demand2)[0]

    correlations = correlations.drop(columns=[1,2,4,6,9,10,12,13,14,15,16])
    for b_id in range(1, n_buildings + 1):
        b_correlations = np.array(correlations.iloc[b_id-1][:-1])
        correlations.iloc[b_id-1]['Median'] = np.median(b_correlations)

    correlations.to_csv('../building_correlations.csv')
