import numpy as np
import pandas as pd
import os
from glob import glob

from scipy.stats import pearsonr

import matplotlib.pyplot as plt

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

PATH = "/Users/kristina/Documents/Studium/Informatik M.Sc. - Tü/SoSe 2023/social-rl/experiments"
EXT = "kpis_*.csv"

if __name__ == '__main__':
    all_csv_files = [file
                     for path, subdir, files in os.walk(PATH)
                     for file in glob(os.path.join(path, EXT))]

    fossil_kpi = []
    energy_kpi = []
    n = 0
    for file in all_csv_files:
        df = pd.read_csv(file)
        df = df.set_index('kpi')
        df = df[df['level'] == 'district']

        for env_id in df['env_id'].unique():
            try:
                fossil_kpi.append(df[(df['env_id'] == env_id)].loc['fossil_energy_consumption', 'value'])
                energy_kpi.append(df[(df['env_id'] == env_id)].loc['electricity_consumption', 'value'])
                n += 1
            except:
                pass

    correlation = pearsonr(fossil_kpi, energy_kpi)[0]
    print('Correlation: ', correlation)
    print('Number of experiments: ', n)

    training_buildings_rbc = 1.066
    eval_buildings_rbc = 1.068

    training_buildings_sac = 0.929
    eval_buildings_sac = 0.911

    training_buildings_sac = [0.921, 0.946, 0.939, 0.915, 0.93, 0.95]
    eval_buildings_sac = [0.91, 0.901, 0.936, 0.953, 0.868, 0.941]

    std_training = np.std(training_buildings_sac)
    std_eval = np.std(eval_buildings_sac)

    print(std_training)
    print(std_eval)

    # Provided data
    training_buildings_sac = np.array([0.921, 0.946, 0.939, 0.915, 0.93, 0.95])
    eval_buildings_sac = np.array([0.91, 0.901, 0.936, 0.953, 0.868, 0.941])

    # Prepare the positions for each group on the x-axis
    x_train = np.ones_like(training_buildings_sac)
    x_eval = 1.25 * np.ones_like(eval_buildings_sac)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 3))

    # Plot individual data points
    ax.scatter(training_buildings_sac, x_train, alpha=0.6, # label='Training Buildings SAC Best',
               marker='x', zorder=2, s=100)
    ax.scatter(eval_buildings_sac, x_eval, alpha=0.6, # label='Eval Buildings SAC Best',
               marker='x', zorder=2, s=100)

    # Calculate and plot the mean and standard deviation for the training group
    mean_training = np.mean(training_buildings_sac)
    std_training = np.std(training_buildings_sac)
    ax.errorbar(mean_training, 1, xerr=std_training, fmt='o', color='tab:blue', capsize=8, markersize=10,
                label=r'Mean \& Std Dev (Training)', zorder=3)

    # Calculate and plot the mean and standard deviation for the evaluation group
    mean_eval = np.mean(eval_buildings_sac)
    std_eval = np.std(eval_buildings_sac)
    ax.errorbar(mean_eval, 1.25, xerr=std_eval, fmt='o', color='tab:orange', capsize=8, markersize=10,
                label=r'Mean \& Std Dev (Eval)', zorder=3)

    # Add labels and title
    ax.set_yticks([1, 1.25], ['Training Buildings', 'Evaluation Buildings'], fontsize=17)
    ax.set_xlabel('Energy Consumption (SAC)', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Add a legend and grid
    #ax.legend(fontsize=17)
    mean_line = plt.errorbar(0.85, 1.1, xerr=0.01, fmt='o', color='black', ecolor='black', capsize=8, markersize=10, elinewidth=2)
    ax.legend([mean_line], ['Mean and Standard Deviation'], fontsize=15, loc='upper right')
    mean_line.remove()

    ax.set_ylim([0.85,1.45])
    ax.grid(axis='x', zorder=1)

    plt.tight_layout()

    # Display the plot
    plt.show()

    fig.savefig('building_sac.pdf')