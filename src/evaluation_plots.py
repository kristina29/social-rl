import glob
from argparse import ArgumentParser

import pandas as pd
import matplotlib.pyplot as plt

EXPERIMENT_BASE_DIR = '../experiments/'

parser = ArgumentParser()
parser.add_argument('--agent', help='Name of the directory of chosen agent', default='DB2_SAC')
parser.add_argument('--dirs', help='Names of the directories of the experiments to compare', nargs='+')
opts = parser.parse_args()

if __name__ == '__main__':
    experiment_dirs = parser.dirs

    kpis = []
    for dir in experiment_dirs:
        kpi_filenames = glob.glob(f'{EXPERIMENT_BASE_DIR}{parser.agent}/{dir}/kpis_*.csv')

        if len(kpi_filenames) > 1:
            raise ValueError(f'More than one KPI csv file found in {dir}')

        kpis.append(pd.read_csv(kpi_filenames[0]))





