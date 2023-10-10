import copy
import time

from citylearn.agents.marlisa import MARLISA

from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from citylearn.utilities import get_active_parts
from nonsocialrl import train_rbc
from options import parseOptions_nonsocial
from utils import set_schema_buildings, set_active_observations, save_results


def train(dataset_name, random_seed, building_count, episodes, active_observations, batch_size, discount,
          autotune_entropy, clip_gradient, kaiming_initialization, l2_loss, exclude_rbc,
          building_ids, store_agents, end_exploration_t):
    # Train SAC agent on defined dataset
    # Workflow strongly based on the citylearn_ccai_tutorial

    # load data
    schema = DataSet.get_schema(dataset_name)

    # TODO: DATA EXPLORATION

    # Data Preprocessing
    schema = preprocessing(schema, building_count, random_seed, active_observations, building_ids=building_ids)

    all_envs = {}
    all_losses = {}
    all_rewards = {}
    all_agents = {}
    all_eval_results = {}
    # Train rule-based control (RBC) agent for comparison
    if not exclude_rbc:
        all_envs['RBC'], all_agents['RBC'] = train_rbc(schema=schema, episodes=episodes)

    # Train MARLISA agent
    all_envs['MARLISA'], all_losses['MARLISA'], all_rewards['MARLISA'], all_eval_results['MARLISA'], \
    all_agents['MARLISA'], all_envs['MARLISA Best'] = \
        train_marlisa(schema=schema, episodes=episodes, random_seed=random_seed, batch_size=batch_size,
                      discount=discount,
                      autotune_entropy=autotune_entropy, clip_gradient=clip_gradient,
                      kaiming_initialization=kaiming_initialization, l2_loss=l2_loss,
                      end_exploration_t=end_exploration_t)

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


def train_marlisa(schema, episodes, random_seed, batch_size, discount, autotune_entropy, clip_gradient,
                  kaiming_initialization, l2_loss, end_exploration_t):
    env = CityLearnEnv(schema)
    marlisa_model = MARLISA(env=env, seed=random_seed, batch_size=batch_size, autotune_entropy=autotune_entropy,
                            clip_gradient=clip_gradient, kaiming_initialization=kaiming_initialization, l2_loss=l2_loss,
                            discount=discount, end_exploration_time_step=end_exploration_t,
                            start_regression_time_step=5500)
    losses, rewards, eval_results, best_state = marlisa_model.learn(episodes=episodes, deterministic_finish=True)

    best_state_env = copy.deepcopy(marlisa_model.env)
    eval_observations = best_state_env.reset()

    while not best_state_env.done:
        actions = best_state.predict(eval_observations, deterministic=True)
        eval_observations, eval_rewards, _, _ = best_state_env.step(actions)

    print('MARLISA model trained!')

    return env, losses, rewards, eval_results, marlisa_model, best_state_env


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
    building_ids = opts.building_ids
    store_agents = opts.store_agents
    end_exploration_t = opts.end_exploration_t

    if False:
        DATASET_NAME = 'nnb_limitobs1_marlisa'
        exclude_rbc = 0
        exclude_tql = 1
        building_count = 2
        episodes = 2
        seed = 2
        autotune_entropy = True
        discount = 0.99
        building_ids = None
        active_observations = ['hour']  # ['solar_generation', 'electrical_storage_soc', 'non_shiftable_load']  # , 'electricity_pricing', 'electricity_pricing_predicted_6h',
        # '#electricity_pricing_predicted_12h', 'electricity_pricing_predicted_24h']
        batch_size = 256
        clip_gradient = False
        store_agents = False
        kaiming_initialization = False
        l2_loss = False
        end_exploration_t = 9000

    train(dataset_name=DATASET_NAME, random_seed=seed, building_count=building_count, episodes=episodes,
          active_observations=active_observations, batch_size=batch_size, discount=discount,
          autotune_entropy=autotune_entropy, clip_gradient=clip_gradient, kaiming_initialization=kaiming_initialization,
          l2_loss=l2_loss, exclude_rbc=exclude_rbc, building_ids=building_ids,
          store_agents=store_agents, end_exploration_t=end_exploration_t)

    # get the end time
    et = time.time()

    # print the execution time
    elapsed_time = et - st
    print("")
    print('Execution time:', round(elapsed_time / 60), 'minutes')
