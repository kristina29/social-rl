import pickle

from matplotlib import pyplot as plt

from utils import plot_used_pv_share, plot_renewable_share, plot_fossil_consumption

if __name__ == '__main__':
    file = '../experiments/SAC_DB2/envs_20231109T153440.pkl'

    with open(file, 'rb') as file:
        envs = pickle.load(file)

    plot_used_pv_share(envs)
    plot_renewable_share(envs, grid=True)
    plot_renewable_share(envs)
    plot_fossil_consumption(envs)

    plt.show()