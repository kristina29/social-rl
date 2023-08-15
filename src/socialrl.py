import time

from datetime import datetime

from citylearn.agents.db2_sac import SACDB2
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from citylearn.utilities import get_active_parts
from options import parseOptions_social
from utils import set_schema_buildings, set_active_observations, plot_simulation_summary, set_schema_demonstrators, \
    save_kpis, save_results
from nonsocialrl import train_tql, train_rbc, train_sac


def train(dataset_name, random_seed, building_count, demonstrators_count, episodes, active_observations, batch_size,
          autotune_entropy, clip_gradient, kaiming_initialization, exclude_tql, exclude_rbc, exclude_sac):
    # Train SAC agent on defined dataset
    # Workflow strongly based on the citylearn_ccai_tutorial

    # load data
    schema = DataSet.get_schema(dataset_name)

    # TODO: DATA EXPLORATION

    # Data Preprocessing
    schema = preprocessing(schema, building_count, demonstrators_count, random_seed, active_observations)

    all_envs = {}
    all_losses = {}
    all_rewards = {}
    # Train rule-based control (RBC) agent for comparison
    if not exclude_rbc:
        all_envs['RBC'] = train_rbc(schema, episodes)

    # Train tabular Q-Learning (TQL) agent for comparison
    if not exclude_tql:
        all_envs['TQL'] = train_tql(schema, active_observations, episodes)

    # Train soft actor-critic (SAC) agent for comparison
    if not exclude_sac:
        all_envs['SAC'], all_losses['SAC'], all_rewards['SAC'] = train_sac(schema, episodes, random_seed, batch_size,
                                                                           autotune_entropy, clip_gradient,
                                                                           kaiming_initialization)

    # Train SAC agent with decision-biasing
    all_envs['SAC_DB2'], all_losses['SAC_DB2'], all_rewards['SAC_DB2'] = train_sacdb2(schema, episodes, random_seed,
                                                                                      batch_size, autotune_entropy,
                                                                                      clip_gradient,
                                                                                      kaiming_initialization)

    save_results(all_envs, all_losses, all_rewards)


def preprocessing(schema, building_count, demonstrators_count, random_seed, active_observations):
    if building_count is not None:
        schema, buildings = set_schema_buildings(schema, building_count, random_seed)
        print('Selected buildings:', buildings)
    if demonstrators_count is not None and demonstrators_count <= building_count:
        schema, demonstrators = set_schema_demonstrators(schema, demonstrators_count, random_seed)
        print('Selected demonstrators:', demonstrators)
    elif demonstrators_count is not None:
        raise ValueError('Number of demonstrators is higher than number of buildings.')
    else:
        raise ValueError('Number of demonstrators is not defined. This is mandatory for social learning.')
    if active_observations is not None:
        schema, active_observations = set_active_observations(schema, active_observations)
    else:
        active_observations = get_active_parts(schema, 'observations')
    print(f'Active observations:', active_observations)

    return schema


def train_sacdb2(schema, episodes, random_seed, batch_size, autotune_entropy, clip_gradient, kaiming_initialization):
    env = CityLearnEnv(schema)
    sacdb2_model = SACDB2(env=env, seed=random_seed, batch_size=batch_size, autotune_entropy=autotune_entropy,
                          clip_gradient=clip_gradient, kaiming_initialization=kaiming_initialization)
    losses, rewards = sacdb2_model.learn(episodes=episodes, deterministic_finish=True)

    print('SAC DB2 model trained!')

    return env, losses, rewards


if __name__ == '__main__':
    st = time.time()

    opts = parseOptions_social()

    DATASET_NAME = opts.schema
    seed = opts.seed
    building_count = opts.buildings
    demonstrators_count = opts.demonstrators
    episodes = opts.episodes
    exclude_tql = opts.exclude_tql
    exclude_rbc = opts.exclude_rbc
    exclude_sac = opts.exclude_sac
    active_observations = opts.observations
    batch_size = opts.batch
    autotune_entropy = opts.autotune
    clip_gradient = opts.clipgradient
    kaiming_initialization = opts.kaiming

    if True:
        DATASET_NAME = 'nydata'
        exclude_rbc = 1
        exclude_tql = 1
        exclude_sac = 0
        demonstrators_count = 1
        building_count = 2
        episodes = 2
        seed = 2
        active_observations = ['renewable_energy_produced']
        batch_size = 256
        autotune_entropy = False
        clip_gradient = False
        kaiming_initialization = False

    train(DATASET_NAME, seed, building_count, demonstrators_count, episodes, active_observations, batch_size,
          autotune_entropy, clip_gradient, kaiming_initialization, exclude_tql, exclude_rbc, exclude_sac)

    # get the end time
    et = time.time()

    # print the execution time
    elapsed_time = et - st
    print("")
    print('Execution time:', round(elapsed_time / 60), 'minutes')
