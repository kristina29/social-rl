import sys
import time

from datetime import datetime

from matplotlib import pyplot as plt
import numpy as np

from citylearn.agents.q_learning import TabularQLearning
from citylearn.agents.rbc import OptimizedRBC
from citylearn.agents.sac import SAC
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from citylearn.utilities import get_active_parts
from citylearn.wrappers import TabularQLearningWrapper
from options import parseOptions_nonsocial
from utils import set_schema_buildings, set_active_observations, plot_simulation_summary, save_kpis


def train(dataset_name, random_seed, building_count, episodes, active_observations, batch_size, exclude_tql,
          exclude_rbc):
    # Train SAC agent on defined dataset
    # Workflow strongly based on the citylearn_ccai_tutorial

    # load data
    schema = DataSet.get_schema(dataset_name)

    # TODO: DATA EXPLORATION

    # Data Preprocessing
    schema = preprocessing(schema, building_count, random_seed, active_observations)

    all_envs = {}
    all_losses = {}
    all_rewards = {}
    # Train rule-based control (RBC) agent for comparison
    if not exclude_rbc:
        all_envs['RBC'] = train_rbc(schema, episodes, batch_size)

    # Train tabular Q-Learning (TQL) agent for comparison
    if not exclude_tql:
        all_envs['TQL'] = train_tql(schema, active_observations, episodes)

    # Train soft actor-critic (SAC) agent
    all_envs['SAC'], all_losses['SAC'], all_rewards['SAC'] = train_sac(schema, episodes, random_seed, batch_size)
    print('SAC model trained!')

    # plot summary and compare with other control results
    filename = f'plots_{datetime.now().strftime("%Y%m%dT%H%M%S")}'
    plot_simulation_summary(all_envs, all_losses, all_rewards, filename)

    # save KPIs as csv
    filename = f'kpis_{datetime.now().strftime("%Y%m%dT%H%M%S")}.csv'
    save_kpis(all_envs, filename)
    print(f'KPIs saved to {filename}')



def preprocessing(schema, building_count, random_seed, active_observations):
    if building_count is not None:
        schema, buildings = set_schema_buildings(schema, building_count, random_seed)
        print('Selected buildings:', buildings)
    if active_observations is not None:
        schema, active_observations = set_active_observations(schema, active_observations)
    else:
        active_observations = get_active_parts(schema, 'observations')
    print(f'Active observations:', active_observations)

    return schema


def get_bins(schema, key, active_parts=None):
    if active_parts is None:
        active_parts = get_active_parts(schema, key)

    bins = {}
    all_parts = schema[key]
    for part in all_parts:
        bins[part] = all_parts[part]['bins'] if part in active_parts else None

    return bins


def train_rbc(schema, episodes):
    env = CityLearnEnv(schema)
    rbc_model = OptimizedRBC(env)
    rbc_model.learn(episodes=episodes)

    print('RBC model trained!')

    return env


def train_tql(schema, active_observations, episodes):
    env = CityLearnEnv(schema)

    # discretize active observations and actions
    observation_bins = get_bins(schema, 'observations', active_observations)
    action_bins = get_bins(schema, 'actions')

    # initialize list of bin sizes where each building
    # has a dictionary in the list definining its bin sizes
    observation_bin_sizes = []
    action_bin_sizes = []

    for _ in env.buildings:
        observation_bin_sizes.append(observation_bins)
        action_bin_sizes.append(action_bins)

    env = TabularQLearningWrapper(env.unwrapped,
                                  observation_bin_sizes=observation_bin_sizes,
                                  action_bin_sizes=action_bin_sizes)

    tql_model = TabularQLearning(env)
    tql_model.learn(episodes=episodes)

    print('TQL model trained!')

    return env


def train_sac(schema, episodes, random_seed, batch_size):
    env = CityLearnEnv(schema)
    sac_model = SAC(env=env, seed=random_seed, batch_size=batch_size)
    losses, rewards = sac_model.learn(episodes=episodes, deterministic_finish=True)

    return env, losses, rewards


if __name__ == '__main__':
    st = time.time()

    opts = parseOptions_nonsocial()

    DATASET_NAME = opts.schema
    seed = opts.seed
    building_count = opts.buildings
    episodes = opts.episodes
    exclude_tql = opts.exclude_tql
    exclude_rbc = opts.exclude_rbc
    active_observations = opts.observations
    batch_size = opts.batch

    if False:
        DATASET_NAME = 'nydata'
        exclude_rbc = 1
        exclude_tql = 1
        building_count = 2
        episodes = 2
        seed = 2
        active_observations = ['renewable_energy_produced']
        batch_size = 256

    print(batch_size)
    train(DATASET_NAME, seed, building_count, episodes, active_observations, batch_size, exclude_tql, exclude_rbc)

    # get the end time
    et = time.time()

    # print the execution time
    elapsed_time = et - st
    print("")
    print('Execution time:', round(elapsed_time / 60), 'minutes')
