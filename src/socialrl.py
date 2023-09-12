import time

from citylearn.agents.db2_sac import SACDB2
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from citylearn.utilities import get_active_parts
from options import parseOptions_social
from utils import set_schema_buildings, set_active_observations, set_schema_demonstrators, save_results
from nonsocialrl import train_tql, train_rbc, train_sac


def train(dataset_name, random_seed, building_count, demonstrators_count, episodes, discount, active_observations,
          batch_size, autotune_entropy, clip_gradient, kaiming_initialization, l2_loss, exclude_tql, exclude_rbc,
          exclude_sac, mode, imitation_lr, building_id, store_agents):
    # Train SAC agent on defined dataset
    # Workflow strongly based on the citylearn_ccai_tutorial

    # load data
    schema = DataSet.get_schema(dataset_name)

    # TODO: DATA EXPLORATION

    # Data Preprocessing
    schema = preprocessing(schema, building_count, demonstrators_count, random_seed, active_observations, building_id)

    all_envs = {}
    all_agents = {}
    all_losses = {}
    all_rewards = {}
    all_eval_results = {}
    # Train rule-based control (RBC) agent for comparison
    if not exclude_rbc:
        all_envs['RBC'], all_agents['RBC'] = train_rbc(schema, episodes)

    # Train tabular Q-Learning (TQL) agent for comparison
    if not exclude_tql:
        all_envs['TQL'], all_agents['TQL'] = train_tql(schema, active_observations, episodes)

    # Train soft actor-critic (SAC) agent for comparison
    if not exclude_sac:
        all_envs['SAC'], all_losses['SAC'], all_rewards['SAC'], all_eval_results['SAC'], all_agents['SAC'] = \
            train_sac(schema=schema, episodes=episodes, random_seed=random_seed, batch_size=batch_size,
                      discount=discount, autotune_entropy=autotune_entropy, clip_gradient=clip_gradient,
                      kaiming_initialization=kaiming_initialization, l2_loss=l2_loss)

    # Train SAC agent with decision-biasing
    all_envs['SAC_DB2'], all_losses['SAC_DB2'], all_rewards['SAC_DB2'], all_eval_results['SAC_DB2'], \
        all_agents['SAC_DB2'] = \
        train_sacdb2(schema=schema, episodes=episodes, random_seed=random_seed, batch_size=batch_size,
                     discount=discount, autotune_entropy=autotune_entropy, clip_gradient=clip_gradient,
                     kaiming_initialization= kaiming_initialization, l2_loss=l2_loss, mode=mode, 
                     imitation_lr=imitation_lr)

    save_results(all_envs, all_losses, all_rewards, all_eval_results, agents=all_agents, store_agents=store_agents)


def preprocessing(schema, building_count, demonstrators_count, random_seed, active_observations, building_id):
    if building_id is not None:
        schema, buildings = set_schema_buildings(schema, building_id=building_id)
        print('Selected buildings:', buildings)
    elif building_count is not None:
        schema, buildings = set_schema_buildings(schema, count=building_count, seed=random_seed)
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


def train_sacdb2(schema, episodes, random_seed, batch_size, discount, autotune_entropy, clip_gradient,
                 kaiming_initialization, l2_loss, mode, imitation_lr):
    env = CityLearnEnv(schema)
    sacdb2_model = SACDB2(env=env, seed=random_seed, batch_size=batch_size, autotune_entropy=autotune_entropy,
                          clip_gradient=clip_gradient, kaiming_initialization=kaiming_initialization, l2_loss=l2_loss,
                          discount=discount, mode=mode, imitation_lr=imitation_lr)#,
                          #start_training_time_step=1, end_exploration_time_step=7000)
    losses, rewards, eval_results = sacdb2_model.learn(episodes=episodes, deterministic_finish=True)

    print('SAC DB2 model trained!')

    return env, losses, rewards, eval_results, sacdb2_model


if __name__ == '__main__':
    st = time.time()

    opts = parseOptions_social()

    DATASET_NAME = opts.schema
    seed = opts.seed
    building_count = opts.buildings
    demonstrators_count = opts.demonstrators
    episodes = opts.episodes
    discount = opts.discount
    exclude_tql = opts.exclude_tql
    exclude_rbc = opts.exclude_rbc
    exclude_sac = opts.exclude_sac
    active_observations = opts.observations
    batch_size = opts.batch
    autotune_entropy = opts.autotune
    clip_gradient = opts.clipgradient
    kaiming_initialization = opts.kaiming
    l2_loss = opts.l2_loss
    mode = opts.mode
    imitation_lr = opts.ir
    building_id = opts.building_id
    store_agents = opts.store_agents

    if False:
        DATASET_NAME = 'nydata'
        exclude_rbc = 0
        exclude_tql = 1
        exclude_sac = 0
        demonstrators_count = 1
        building_count = 2
        episodes = 2
        discount = 0.99
        seed = 2
        active_observations = ['renewable_energy_produced']
        batch_size = 256
        imitation_lr = 0.01
        mode = 1
        autotune_entropy = True
        clip_gradient = False
        kaiming_initialization = False
        l2_loss = True
        building_id = None
        store_agents = False

    train(dataset_name=DATASET_NAME, random_seed=seed, building_count=building_count,
          demonstrators_count=demonstrators_count, episodes=episodes, discount=discount,
          active_observations=active_observations, batch_size=batch_size, autotune_entropy=autotune_entropy,
          clip_gradient=clip_gradient, kaiming_initialization=kaiming_initialization, l2_loss=l2_loss,
          exclude_tql=exclude_tql, exclude_rbc=exclude_rbc, exclude_sac=exclude_sac, mode=mode, 
          imitation_lr=imitation_lr, building_id=building_id, store_agents=store_agents)

    # get the end time
    et = time.time()

    # print the execution time
    elapsed_time = et - st
    print("")
    print('Execution time:', round(elapsed_time / 60), 'minutes')
