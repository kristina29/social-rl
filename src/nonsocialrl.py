import copy
import time

from citylearn.agents.q_learning import TabularQLearning
from citylearn.agents.rbc import OptimizedRBC
from citylearn.agents.sac import SAC
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from citylearn.utilities import get_active_parts
from citylearn.wrappers import TabularQLearningWrapper
from options import parseOptions_nonsocial
from utils import set_schema_buildings, set_active_observations, save_results, save_transitions_to, get_best_env


def train(dataset_name, random_seed, building_count, episodes, active_observations, batch_size, discount,
          autotune_entropy, clip_gradient, kaiming_initialization, l2_loss, include_tql, include_rbc,
          building_ids, store_agents, end_exploration_t, save_transitions):
    # Train SAC agent on defined dataset
    # Workflow strongly based on the citylearn_ccai_tutorial

    # load data
    schema = DataSet.get_schema(dataset_name)

    # Data Preprocessing
    schema = preprocessing(schema, building_count, random_seed, active_observations, building_ids=building_ids)

    all_envs = {}
    all_losses = {}
    all_rewards = {}
    all_agents = {}
    all_eval_results = {}
    # Train rule-based control (RBC) agent for comparison
    if include_rbc:
        all_envs['RBC'], all_agents['RBC'] = train_rbc(schema=schema, episodes=episodes)

    # Train tabular Q-Learning (TQL) agent for comparison
    if include_tql:
        all_envs['TQL'], all_agents['TQL'] = train_tql(schema=schema, active_observations=active_observations,
                                                       episodes=episodes)

    # Train soft actor-critic (SAC) agent
    all_envs['SAC'], all_losses['SAC'], all_rewards['SAC'], all_eval_results['SAC'], all_agents['SAC'],\
        all_envs['SAC Best'] = \
        train_sac(schema=schema, episodes=episodes, random_seed=random_seed, batch_size=batch_size, discount=discount,
                  autotune_entropy=autotune_entropy, clip_gradient=clip_gradient,
                  kaiming_initialization=kaiming_initialization, l2_loss=l2_loss, end_exploration_t=end_exploration_t,
                  save_transitions=save_transitions)

    save_results(all_envs, all_losses, all_rewards, all_eval_results, agents=all_agents, store_agents=store_agents)


def preprocessing(schema, building_count, random_seed, active_observations, building_ids):
    if building_ids is not None:
        schema, buildings = set_schema_buildings(schema, building_ids_to_include=building_ids)
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
              kaiming_initialization, l2_loss, end_exploration_t, save_transitions):
    env = CityLearnEnv(schema)
    sac_model = SAC(env=env, seed=random_seed, batch_size=batch_size, autotune_entropy=autotune_entropy,
                    clip_gradient=clip_gradient, kaiming_initialization=kaiming_initialization, l2_loss=l2_loss,
                    discount=discount, end_exploration_time_step=end_exploration_t)
    losses, rewards, eval_results, best_state = sac_model.learn(episodes=episodes, deterministic_finish=True)

    best_state_env = get_best_env(sac_model, best_state)

    print('SAC model trained!')

    if save_transitions:
        save_transitions_to(env, sac_model, 'SAC')

    return env, losses, rewards, eval_results, sac_model, best_state_env


if __name__ == '__main__':
    st = time.time()

    opts = parseOptions_nonsocial()

    DATASET_NAME = opts.schema
    seed = opts.seed
    building_count = opts.buildings
    episodes = opts.episodes
    discount = opts.discount
    include_tql = opts.include_tql
    include_rbc = opts.include_rbc
    active_observations = opts.observations
    batch_size = opts.batch
    autotune_entropy = opts.autotune
    clip_gradient = opts.clipgradient
    kaiming_initialization = opts.kaiming
    l2_loss = opts.l2_loss
    building_ids = opts.building_ids
    store_agents = opts.store_agents
    end_exploration_t = opts.end_exploration_t
    save_transitions = opts.save_transitions

    train(dataset_name=DATASET_NAME, random_seed=seed, building_count=building_count, episodes=episodes,
          active_observations=active_observations, batch_size=batch_size, discount=discount,
          autotune_entropy=autotune_entropy, clip_gradient=clip_gradient, kaiming_initialization=kaiming_initialization,
          l2_loss=l2_loss, include_tql=include_tql, include_rbc=include_rbc, building_ids=building_ids,
          store_agents=store_agents, end_exploration_t=end_exploration_t, save_transitions=save_transitions)

    # get the end time
    et = time.time()

    # print the execution time
    elapsed_time = et - st
    print("")
    print('Execution time:', round(elapsed_time / 60), 'minutes')
