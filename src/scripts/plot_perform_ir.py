import glob

import matplotlib.pyplot as plt
import pandas as pd


def plot(dirs):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })

    fossil_energy = []
    irs = []

    fossil_energy2 = []
    irs2 = []

    for ir in [0.0001, 0.001, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8]:
        file = glob.glob(f'{dirs[0]}/ir{ir}/kpis_*.csv')[0]
        kpis = pd.read_csv(file)
        kpis = kpis.set_index('kpi')
        kpis = kpis[(kpis['env_id'] == 'SAC_DB2Value Best') & (kpis['level'] == 'district')]
        v = round(kpis.loc['fossil_energy_consumption', 'net_value'] /
                  kpis.loc['fossil_energy_consumption', 'net_value_without_storage'], 3)

        irs.append(ir)
        fossil_energy.append(v)

    for ir in [0.0001, 0.001, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8]:
        file = glob.glob(f'{dirs[1]}/ir{ir}/kpis_*.csv')[0]
        kpis = pd.read_csv(file)
        kpis = kpis.set_index('kpi')
        kpis = kpis[(kpis['env_id'] == 'SAC_DB2Value Best') & (kpis['level'] == 'district')]
        v = round(kpis.loc['fossil_energy_consumption', 'net_value'] /
                  kpis.loc['fossil_energy_consumption', 'net_value_without_storage'], 3)

        irs2.append(ir)
        fossil_energy2.append(v)

    fig, ax = plt.subplots()
    ax.plot(irs, fossil_energy)
    ax.scatter(irs, fossil_energy)
    ax.plot(irs2, fossil_energy2)
    ax.scatter(irs2, fossil_energy2)
    plt.show()


if __name__ == '__main__':
    paths = ['../experiments/SAC_DB2Value/2_demo_b6',
             '../experiments/SAC_DB2Value/4_demo_b6_policyupdate']

    plot(paths)