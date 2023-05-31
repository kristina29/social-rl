import sys

from datetime import datetime

from citylearn.agents.db2_sac import SACDB2
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from options import parseOptions_social
from utils import set_schema_buildings, set_active_observations, plot_simulation_summary, set_schema_demonstrators, \
    get_active_parts
from nonsocialrl import train_tql, train_rbc, train_sac


def train(dataset_name, random_seed, building_count, demonstrators_count, episodes, active_observations, exclude_tql,
          exclude_rbc, exclude_sac):
    # Train SAC agent on defined dataset
    # Workflow strongly based on the citylearn_ccai_tutorial

    # load data
    schema = DataSet.get_schema(dataset_name)

    # TODO: DATA EXPLORATION

    # Data Preprocessing
    schema = preprocessing(schema, building_count, demonstrators_count, random_seed, active_observations)

    all_envs = {}
    # Train rule-based control (RBC) agent for comparison
    if not exclude_rbc:
        all_envs['RBC'] = train_rbc(schema, episodes)

    # Train tabular Q-Learning (TQL) agent for comparison
    if not exclude_tql:
        all_envs['TQL'] = train_tql(schema, active_observations, episodes)

    # Train soft actor-critic (SAC) agent for comparison
    if not exclude_sac:
        all_envs['SAC'] = train_sac(schema, episodes, random_seed)

    # Train SAC agent with decision-biasing
    all_envs['SAC_DB2'] = train_sacdb2(schema, episodes, random_seed)

    # plot summary and compare with other control results
    filename = "plots_" + datetime.now().strftime("%Y%m%dT%H%M%S")
    plot_simulation_summary(all_envs, filename)


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
        active_observations = get_active_parts(schema)
    print(f'Active observations:', active_observations)

    return schema


def train_sacdb2(schema, episodes, random_seed):
    env = CityLearnEnv(schema)
    sacdb2_model = SACDB2(env=env, seed=random_seed)
    sacdb2_model.learn(episodes=episodes, deterministic_finish=True)

    print('SAC DB2 model trained!')

    return env


if __name__ == '__main__':
    #  BINS DEFINED IN SCHEMA 

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

    # only when used in pycharm for testing
    if len(sys.argv) == 10:
        DATASET_NAME = sys.argv[1]
        seed = int(sys.argv[2])
        building_count = int(sys.argv[3])
        demonstrators_count = int(sys.argv[4])
        episodes = int(sys.argv[5])
        exclude_tql = bool(int(sys.argv[6]))
        exclude_rbc = bool(int(sys.argv[7]))
        exclude_sac = bool(int(sys.argv[8]))
        active_observations = [sys.argv[9]]

    train(DATASET_NAME, seed, building_count, demonstrators_count, episodes, active_observations, exclude_tql,
          exclude_rbc, exclude_sac)
