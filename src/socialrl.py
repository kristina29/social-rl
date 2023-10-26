import copy
import pickle
import time

from citylearn.agents.db2_sac import SACDB2
from citylearn.agents.db2_value_sac import SACDB2VALUE
from citylearn.agents.dpb_sac import PRBSAC
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from citylearn.utilities import get_active_parts
from options import parseOptions_social
from utils import set_schema_buildings, set_active_observations, set_schema_demonstrators, save_results, \
    save_transitions_to
from nonsocialrl import train_tql, train_rbc, train_sac


def train(dataset_name, random_seed, building_count, demonstrators_count, episodes, discount, active_observations,
          batch_size, autotune_entropy, clip_gradient, kaiming_initialization, l2_loss, exclude_tql, exclude_rbc,
          exclude_sac, exclude_sacdb2, exclude_sacdb2value, mode, imitation_lr, building_ids, store_agents,
          pretrained_demonstrator, demo_transitions, deterministic_demo, extra_policy_update, end_exploration_t,
          save_transitions):
    # Train SAC agent on defined dataset
    # Workflow strongly based on the citylearn_ccai_tutorial

    # load data
    schema = DataSet.get_schema(dataset_name)

    # TODO: DATA EXPLORATION

    # Data Preprocessing
    schema = preprocessing(schema, building_count, demonstrators_count, random_seed, active_observations, building_ids)

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
        all_envs['SAC'], all_losses['SAC'], all_rewards['SAC'], all_eval_results['SAC'], all_agents['SAC'],\
            all_envs['SAC Best'] = \
            train_sac(schema=schema, episodes=episodes, random_seed=random_seed, batch_size=batch_size,
                      discount=discount, autotune_entropy=autotune_entropy, clip_gradient=clip_gradient,
                      kaiming_initialization=kaiming_initialization, l2_loss=l2_loss,
                      end_exploration_t=end_exploration_t, save_transitions=save_transitions)

    # Train SAC agent with decision-biasing
    if not exclude_sacdb2:
        all_envs['SAC_DB2'], all_losses['SAC_DB2'], all_rewards['SAC_DB2'], all_eval_results['SAC_DB2'], \
        all_agents['SAC_DB2'], all_envs['SAC_DB2 Best'] = \
            train_sacdb2(schema=schema, episodes=episodes, random_seed=random_seed, batch_size=batch_size,
                         discount=discount, autotune_entropy=autotune_entropy, clip_gradient=clip_gradient,
                         kaiming_initialization=kaiming_initialization, l2_loss=l2_loss, mode=mode,
                         imitation_lr=imitation_lr, pretrained_demonstrator=pretrained_demonstrator,
                         deterministic_demo=deterministic_demo, end_exploration_t=end_exploration_t,
                         save_transitions=save_transitions)

    # Train SAC agent with decision-biasing on the value function
    if not exclude_sacdb2value:
        all_envs['SAC_DB2Value'], all_losses['SAC_DB2Value'], all_rewards['SAC_DB2Value'], \
        all_eval_results['SAC_DB2Value'], all_agents['SAC_DB2Value'], all_envs['SAC_DB2Value Best'] = \
            train_sacdb2value(schema=schema, episodes=episodes, random_seed=random_seed, batch_size=batch_size,
                              discount=discount, autotune_entropy=autotune_entropy, clip_gradient=clip_gradient,
                              kaiming_initialization=kaiming_initialization, l2_loss=l2_loss,
                              imitation_lr=imitation_lr, pretrained_demonstrator=pretrained_demonstrator,
                              deterministic_demo=deterministic_demo, extra_policy_update=extra_policy_update,
                              end_exploration_t=end_exploration_t, save_transitions=save_transitions)

    # Train SAC agent with demonstrator transitions
    if demo_transitions is not None:
        all_envs['PRB_SAC'], all_losses['PRB_SAC'], all_rewards['PRB_SAC'], all_eval_results['PRB_SAC'], \
        all_agents['PRB_SAC'] = \
            train_prbsac(schema=schema, episodes=episodes, random_seed=random_seed, batch_size=batch_size,
                         discount=discount, autotune_entropy=autotune_entropy, clip_gradient=clip_gradient,
                         kaiming_initialization=kaiming_initialization, demo_transitions=demo_transitions,
                         end_exploration_t=end_exploration_t, l2_loss=l2_loss)

    save_results(all_envs, all_losses, all_rewards, all_eval_results, agents=all_agents, store_agents=store_agents)


def preprocessing(schema, building_count, demonstrators_count, random_seed, active_observations, building_ids):
    if building_ids is not None:
        schema, buildings = set_schema_buildings(schema, building_ids_to_include=building_ids)
        print('Selected buildings:', buildings)
    elif building_count is not None:
        schema, buildings = set_schema_buildings(schema, count=building_count, seed=random_seed)
        print('Selected buildings:', buildings)
    if demonstrators_count is not None and demonstrators_count <= len(buildings):
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
                 kaiming_initialization, l2_loss, mode, imitation_lr, pretrained_demonstrator, deterministic_demo,
                 end_exploration_t, save_transitions):
    if pretrained_demonstrator is not None:
        with open(pretrained_demonstrator, 'rb') as file:
            pretrained_demonstrator = pickle.load(file)

    env = CityLearnEnv(schema)
    sacdb2_model = SACDB2(env=env, seed=random_seed, batch_size=batch_size, autotune_entropy=autotune_entropy,
                          clip_gradient=clip_gradient, kaiming_initialization=kaiming_initialization, l2_loss=l2_loss,
                          discount=discount, mode=mode, imitation_lr=imitation_lr,
                          pretrained_demonstrator=pretrained_demonstrator, deterministic_demo=deterministic_demo,
                          end_exploration_time_step=end_exploration_t)
    losses, rewards, eval_results, best_state = sacdb2_model.learn(episodes=episodes, deterministic_finish=True)

    best_state_env = copy.deepcopy(sacdb2_model.env)
    eval_observations = best_state_env.reset()

    while not best_state_env.done:
        actions = best_state.predict(eval_observations, deterministic=True)
        eval_observations, eval_rewards, _, _ = best_state_env.step(actions)

    print('SAC DB2 model trained!')

    if save_transitions:
        save_transitions_to(env, sacdb2_model, 'SACDB2')

    return env, losses, rewards, eval_results, sacdb2_model, best_state_env


def train_sacdb2value(schema, episodes, random_seed, batch_size, discount, autotune_entropy, clip_gradient,
                      kaiming_initialization, l2_loss, imitation_lr, pretrained_demonstrator, deterministic_demo,
                      extra_policy_update, end_exploration_t, save_transitions):
    if pretrained_demonstrator is not None:
        with open(pretrained_demonstrator, 'rb') as file:
            pretrained_demonstrator = pickle.load(file)

    env = CityLearnEnv(schema)
    sacdb2value_model = SACDB2VALUE(env=env, seed=random_seed, batch_size=batch_size, autotune_entropy=autotune_entropy,
                                    clip_gradient=clip_gradient, kaiming_initialization=kaiming_initialization,
                                    l2_loss=l2_loss,
                                    discount=discount, imitation_lr=imitation_lr,
                                    pretrained_demonstrator=pretrained_demonstrator,
                                    deterministic_demo=deterministic_demo, extra_policy_update=extra_policy_update,
                                    end_exploration_time_step=end_exploration_t)
    losses, rewards, eval_results, best_state = sacdb2value_model.learn(episodes=episodes, deterministic_finish=True)

    best_state_env = copy.deepcopy(sacdb2value_model.env)
    eval_observations = best_state_env.reset()

    while not best_state_env.done:
        actions = best_state.predict(eval_observations, deterministic=True)
        eval_observations, eval_rewards, _, _ = best_state_env.step(actions)

    print('SAC DB2 Value model trained!')

    if save_transitions:
        save_transitions_to(env, sacdb2value_model, 'SACDB2Value')

    return env, losses, rewards, eval_results, sacdb2value_model, best_state_env


def train_prbsac(schema, episodes, random_seed, batch_size, discount, autotune_entropy, clip_gradient,
                 kaiming_initialization, demo_transitions, end_exploration_t, l2_loss):
    env = CityLearnEnv(schema)

    with open(demo_transitions, 'rb') as file:
        demo_transitions = pickle.load(file)

    prbsac_model = PRBSAC(env=env, seed=random_seed, batch_size=batch_size, autotune_entropy=autotune_entropy,
                          clip_gradient=clip_gradient, kaiming_initialization=kaiming_initialization, l2_loss=l2_loss,
                          discount=discount, demonstrator_transitions=demo_transitions,
                          end_exploration_time_step=end_exploration_t)
    losses, rewards, eval_results, best_state = prbsac_model.learn(episodes=episodes, deterministic_finish=True)

    print('PRB SAC model trained!')

    return env, losses, rewards, eval_results, prbsac_model


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
    exclude_sacdb2 = opts.exclude_sacdb2
    exclude_sacdb2value = opts.exclude_sacdb2value
    active_observations = opts.observations
    batch_size = opts.batch
    autotune_entropy = opts.autotune
    clip_gradient = opts.clipgradient
    kaiming_initialization = opts.kaiming
    l2_loss = opts.l2_loss
    mode = opts.mode
    imitation_lr = opts.ir
    building_ids = opts.building_ids
    store_agents = opts.store_agents
    pretrained_demonstrator = opts.pretrained_demonstrator
    demo_transitions = opts.demo_transitions
    deterministic_demo = opts.deterministic_demo
    extra_policy_update = opts.extra_policy_update
    end_exploration_t = opts.end_exploration_t
    save_transitions = opts.save_transitions

    if False:
        DATASET_NAME = 'nydata_new_buildings2'
        exclude_rbc = 1
        exclude_tql = 1
        exclude_sac = 1
        exclude_sacdb2 = 1
        exclude_sacdb2value = 0
        demonstrators_count = 1
        building_count = 2
        episodes = 2
        discount = 0.99
        seed = 2
        active_observations = None  # ['renewable_energy_produced']
        batch_size = 256
        imitation_lr = 0.01
        mode = 1
        autotune_entropy = True
        clip_gradient = False
        kaiming_initialization = False
        l2_loss = False
        building_ids = None
        store_agents = False
        pretrained_demonstrator = None
        demo_transitions = 'sac_transitions_b6.pkl'
        deterministic_demo = False
        extra_policy_update = False
        end_exploration_t = 7000
        save_transitions = False

    if pretrained_demonstrator is not None:
        demonstrators_count = 1
    else:
        demonstrators_count = opts.demonstrators

    print(DATASET_NAME)
    train(dataset_name=DATASET_NAME, random_seed=seed, building_count=building_count,
          demonstrators_count=demonstrators_count, episodes=episodes, discount=discount,
          active_observations=active_observations, batch_size=batch_size, autotune_entropy=autotune_entropy,
          clip_gradient=clip_gradient, kaiming_initialization=kaiming_initialization, l2_loss=l2_loss,
          exclude_tql=exclude_tql, exclude_rbc=exclude_rbc, exclude_sac=exclude_sac, exclude_sacdb2=exclude_sacdb2,
          exclude_sacdb2value=exclude_sacdb2value,
          mode=mode, imitation_lr=imitation_lr, building_ids=building_ids, store_agents=store_agents,
          pretrained_demonstrator=pretrained_demonstrator, demo_transitions=demo_transitions,
          deterministic_demo=deterministic_demo, extra_policy_update=extra_policy_update,
          end_exploration_t=end_exploration_t, save_transitions=save_transitions)

    # get the end time
    et = time.time()

    # print the execution time
    elapsed_time = et - st
    print("")
    print('Execution time:', round(elapsed_time / 60), 'minutes')
