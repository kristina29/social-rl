import gzip
import sys

from datetime import datetime
from citylearn.agents.q_learning import TabularQLearning
from citylearn.agents.rbc import OptimizedRBC
from citylearn.agents.sac import SAC
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from citylearn.wrappers import TabularQLearningWrapper
from options import parseOptions
from utils import set_schema_buildings, set_schema_simulation_period, set_active_observations, plot_simulation_summary


def train(dataset_name, random_seed, building_count, episodes, active_observations, exclude_tql,
          exclude_rbc):
    # Train SAC agent on defined dataset
    # Workflow strongly based on the citylearn_ccai_tutorial

    # load data
    schema = DataSet.get_schema(dataset_name)

    # TODO: DATA EXPLORATION

    # Data Preprocessing
    schema = preprocessing(schema, building_count, random_seed, active_observations)

    all_envs = {}
    # Train rule-based control (RBC) agent for comparison
    if not exclude_rbc:
        all_envs['RBC'] = train_rbc(schema, episodes)

    # Train tabular Q-Learning (TQL) agent for comparison
    if not exclude_tql:
        all_envs['TQL'] = train_tql(schema, active_observations, episodes)

    # Train soft actor-critic (SAC) agent
    all_envs['SAC'] = train_sac(schema, episodes, random_seed)
    print('SAC model trained!')

    # plot summary and compare with other control results
    filename = "plots_" + datetime.now().strftime("%Y%m%dT%H%M%S")
    plot_simulation_summary(all_envs, filename)


def preprocessing(schema, building_count, random_seed, active_observations):
    if building_count is not None:
        schema, buildings = set_schema_buildings(schema, building_count, random_seed)
        print('Selected buildings:', buildings)
    if active_observations is not None:
        schema, active_observations = set_active_observations(schema, active_observations)
    else:
        active_observations = get_active_parts(schema)
    print(f'Active observations:', active_observations)

    return schema


def get_active_parts(schema, key='observations'):
    active_parts = []
    all_parts = schema[key]
    for part in all_parts:
        active_parts.append(part) if all_parts[part]['active'] else None

    return active_parts


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


def train_sac(schema, episodes, random_seed):
    env = CityLearnEnv(schema)
    sac_model = SAC(env=env, seed=random_seed)
    sac_model.learn(episodes=episodes, deterministic_finish=True)

    return env


if __name__ == '__main__':
    #  BINS DEFINED IN SCHEMA 

    opts = parseOptions()

    DATASET_NAME = opts.schema
    seed = opts.seed
    building_count = opts.buildings
    episodes = opts.episodes
    exclude_tql = opts.exclude_tql
    exclude_rbc = opts.exclude_rbc
    active_observations = opts.observations

    # only when used in pycharm for testing
    if len(sys.argv) == 8 and False:
        DATASET_NAME = sys.argv[1]
        seed = int(sys.argv[2])
        building_count = int(sys.argv[3])
        episodes = int(sys.argv[4])
        exclude_tql = bool(int(sys.argv[5]))
        exclude_rbc = bool(int(sys.argv[6]))
        active_observations = [sys.argv[7]]

    train(DATASET_NAME, seed, building_count, episodes, active_observations, exclude_tql, exclude_rbc)
