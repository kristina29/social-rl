import pickle

import os

if __name__ == '__main__':
    with open('../../experiments/SAC_DB2/sac_env.pkl', 'rb') as file:
        env = pickle.load(file)

    env = env