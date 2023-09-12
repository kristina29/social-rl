import pickle
import time

from citylearn.agents.q_learning import TabularQLearning
from citylearn.agents.rbc import OptimizedRBC
from citylearn.agents.sac import SAC
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from citylearn.utilities import get_active_parts
from citylearn.wrappers import TabularQLearningWrapper
from options import parseOptions_nonsocial
from utils import set_schema_buildings, set_active_observations, save_results


def train(dataset_name, random_seed, building_count, episodes, active_observations, batch_size, discount,
          autotune_entropy, clip_gradient, kaiming_initialization, l2_loss, exclude_tql, exclude_rbc,
          building_id, store_agents):
    # Train SAC agent on defined dataset
    # Workflow strongly based on the citylearn_ccai_tutorial

    # load data
    schema = DataSet.get_schema(dataset_name)

    # TODO: DATA EXPLORATION

    # Data Preprocessing
    schema = preprocessing(schema, building_count, random_seed, active_observations, building_id=building_id)

    all_envs = {}
    all_losses = {}
    all_rewards = {}
    all_agents = {}
    all_eval_results = {}
    # Train rule-based control (RBC) agent for comparison
    if not exclude_rbc:
        all_envs['RBC'], all_agents['RBC'] = train_rbc(schema=schema, episodes=episodes)

    # Train tabular Q-Learning (TQL) agent for comparison
    if not exclude_tql:
        all_envs['TQL'], all_agents['TQL'] = train_tql(schema=schema, active_observations=active_observations, episodes=episodes)

    # Train soft actor-critic (SAC) agent
    all_envs['SAC'], all_losses['SAC'], all_rewards['SAC'], all_eval_results['SAC'], all_agents['SAC'] = \
        train_sac(schema=schema, episodes=episodes, random_seed=random_seed, batch_size=batch_size, discount=discount,
                  autotune_entropy=autotune_entropy, clip_gradient=clip_gradient,
                  kaiming_initialization=kaiming_initialization, l2_loss=l2_loss)

    save_results(all_envs, all_losses, all_rewards, all_eval_results, agents=all_agents, store_agents=store_agents)


def preprocessing(schema, building_count, random_seed, active_observations, building_id):
    if building_id is not None:
        schema, buildings = set_schema_buildings(schema, building_id=building_id)
        print('Selected buildings:', buildings)
    elif building_count is not None:
        schema, buildings = set_schema_buildings(schema, count=building_count, seed=random_seed)
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
    schema['observations']['hour']['active'] = True
    env = CityLearnEnv(schema)
    rbc_model = OptimizedRBC(env)
    rbc_model.learn(episodes=episodes)

    print('RBC model trained!')

    return env, rbc_model


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

    return env, tql_model


def train_sac(schema, episodes, random_seed, batch_size, discount, autotune_entropy, clip_gradient,
              kaiming_initialization, l2_loss):
    env = CityLearnEnv(schema)
    sac_model = SAC(env=env, seed=random_seed, batch_size=batch_size, autotune_entropy=autotune_entropy,
                    clip_gradient=clip_gradient, kaiming_initialization=kaiming_initialization, l2_loss=l2_loss,
                    discount=discount) #,
                    #start_training_time_step=1, end_exploration_time_step=14000)
    losses, rewards, eval_results = sac_model.learn(episodes=episodes, deterministic_finish=True, keep_env_history=True)

    filename = f'sac_env.pkl'
    with open(filename, 'wb') as fp:
        pickle.dump(env, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'SAC env saved to {filename}')
    print(f'scp klietz10@134.2.168.52:/mnt/qb/work/ludwig/klietz10/social-rl/{filename}'
          f'experiments/SAC_DB2/{filename}')

    print('SAC model trained!')

    return env, losses, rewards, eval_results, sac_model


if __name__ == '__main__':
    st = time.time()

    opts = parseOptions_nonsocial()

    DATASET_NAME = opts.schema
    seed = opts.seed
    building_count = opts.buildings
    episodes = opts.episodes
    discount = opts.discount
    exclude_tql = opts.exclude_tql
    exclude_rbc = opts.exclude_rbc
    active_observations = opts.observations
    batch_size = opts.batch
    autotune_entropy = opts.autotune
    clip_gradient = opts.clipgradient
    kaiming_initialization = opts.kaiming
    l2_loss = opts.l2_loss
    building_id = opts.building_id
    store_agents = opts.store_agents

    if False:
        DATASET_NAME = 'nydata'
        exclude_rbc = 0
        exclude_tql = 1
        building_count = 2
        episodes = 2
        seed = 2
        autotune_entropy = False
        discount = 0.99
        building_id = 6
        active_observations = ['solar_generation', 'electrical_storage_soc', 'non_shiftable_load']  # , 'electricity_pricing', 'electricity_pricing_predicted_6h',
        #                       'electricity_pricing_predicted_12h', 'electricity_pricing_predicted_24h']
        batch_size = 256
        clip_gradient = False
        store_agents = False
        kaiming_initialization = False
        l2_loss = False

    train(dataset_name=DATASET_NAME, random_seed=seed, building_count=building_count, episodes=episodes,
          active_observations=active_observations, batch_size=batch_size, discount=discount,
          autotune_entropy=autotune_entropy, clip_gradient=clip_gradient, kaiming_initialization=kaiming_initialization,
          l2_loss=l2_loss, exclude_tql=exclude_tql, exclude_rbc=exclude_rbc, building_id=building_id,
          store_agents=store_agents)

    # get the end time
    et = time.time()

    # print the execution time
    elapsed_time = et - st
    print("")
    print('Execution time:', round(elapsed_time / 60), 'minutes')
