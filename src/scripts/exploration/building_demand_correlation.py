import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr

DIR_PATH = 'citylearn/data/nydata_new_buildings2'


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

    #correlations.to_csv('../building_correlations.csv')
    correlations = correlations.astype('float64')

    f = plt.figure(figsize=(18, 15))
    #plt.matshow(np.array(correlations, dtype='float64'), fignum=f.number)
    #cb = plt.colorbar()
    #cb.ax.tick_params(labelsize=14)
    ax = sns.heatmap(
        #np.array(
        correlations
        #, dtype='float64')
        , annot=True, xticklabels=True, yticklabels=True, annot_kws={"fontsize":21})

    buldings = [3,5,7,8,11,17]
    lw = 15
    ax.axvline(6, color='white', lw=lw, zorder=2)

    for b in buldings:
        wanted_label = b
        wanted_index = correlations.index.get_loc(wanted_label)
        x, y, w, h = 0, wanted_index, len(buldings)-0.05, 1
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='black', lw=2.5, clip_on=False, zorder=3))
        x, y, w, h = len(buldings)+0.05, wanted_index, 0.95, 1
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='black', lw=2.5, clip_on=False, zorder=3))

    ax.set_xlabel('Training Building IDs', fontsize=21)
    ax.set_ylabel('Building IDs', fontsize=21)
    ax.tick_params(axis='both', which='major', labelsize=21)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=21)

    plt.tight_layout()

    f.savefig('correlations.pdf')

    plt.show()
