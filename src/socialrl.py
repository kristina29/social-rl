import pickle
import time

from citylearn.agents.db2_sac import SACDEMOPOL
from citylearn.agents.db2_value_sac import SACDEMOQ
from citylearn.agents.dpb_sac import PRBSAC
from citylearn.agents.ddpg import DDPG, PRBDDPG
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from citylearn.utilities import get_active_parts
from options import parseOptions_social
from utils import set_schema_buildings, set_active_observations, set_schema_demonstrators, save_results, \
    save_transitions_to, get_best_env
from nonsocialrl import train_tql, train_rbc, train_sac


def train(dataset_name, random_seed, building_count, demonstrators_count, episodes, discount, active_observations,
          batch_size, autotune_entropy, clip_gradient, kaiming_initialization, l2_loss, include_tql, include_rbc,
          include_sac, include_sacdemopol, include_sacdemoq, mode, imitation_lr, building_ids, store_agents,
          pretrained_demonstrator, demo_transitions, deterministic_demo, extra_policy_update, end_exploration_t,
          save_transitions, ddpg):
    # Train SAC agent on defined dataset
    # Workflow strongly based on the citylearn_ccai_tutorial

    # load data
    schema = DataSet.get_schema(dataset_name)

    # Data Preprocessing
    schema = preprocessing(schema, building_count, demonstrators_count, random_seed, active_observations, building_ids)

    all_envs = {}
    all_agents = {}
    all_losses = {}
    all_rewards = {}
    all_eval_results = {}
    # Train rule-based control (RBC) agent for comparison
    if include_rbc:
        all_envs['RBC'], all_agents['RBC'] = train_rbc(schema, episodes)

    # Train tabular Q-Learning (TQL) agent for comparison
    if include_tql:
        all_envs['TQL'], all_agents['TQL'] = train_tql(schema, active_observations, episodes)

    # Train soft actor-critic (SAC) agent for comparison
    if include_sac:
        all_envs['SAC'], all_losses['SAC'], all_rewards['SAC'], all_eval_results['SAC'], all_agents['SAC'], \
        all_envs['SAC Best'] = \
            train_sac(schema=schema, episodes=episodes, random_seed=random_seed, batch_size=batch_size,
                      discount=discount, autotune_entropy=autotune_entropy, clip_gradient=clip_gradient,
                      kaiming_initialization=kaiming_initialization, l2_loss=l2_loss,
                      end_exploration_t=end_exploration_t, save_transitions=save_transitions)

    # Train SAC agent with decision-biasing
    if include_sacdemopol:
        all_envs['SAC-DemoPol'], all_losses['SAC-DemoPol'], all_rewards['SAC-DemoPol'], all_eval_results['SAC-DemoPol'], \
        all_agents['SAC-DemoPol'], all_envs['SAC-DemoPol Best'] = \
            train_sacdemopol(schema=schema, episodes=episodes, random_seed=random_seed, batch_size=batch_size,
                             discount=discount, autotune_entropy=autotune_entropy, clip_gradient=clip_gradient,
                             kaiming_initialization=kaiming_initialization, l2_loss=l2_loss, mode=mode,
                             imitation_lr=imitation_lr, pretrained_demonstrator=pretrained_demonstrator,
                             deterministic_demo=deterministic_demo, end_exploration_t=end_exploration_t,
                             save_transitions=save_transitions)

    # Train SAC agent with decision-biasing on the value function
    if include_sacdemoq:
        all_envs['SAC-DemoQ'], all_losses['SAC-DemoQ'], all_rewards['SAC-DemoQ'], \
        all_eval_results['SAC-DemoQ'], all_agents['SAC-DemoQ'], all_envs['SAC-DemoQ Best'] = \
            train_sacdemoq(schema=schema, episodes=episodes, random_seed=random_seed, batch_size=batch_size,
                           discount=discount, autotune_entropy=autotune_entropy, clip_gradient=clip_gradient,
                           kaiming_initialization=kaiming_initialization, l2_loss=l2_loss,
                           imitation_lr=imitation_lr, pretrained_demonstrator=pretrained_demonstrator,
                           deterministic_demo=deterministic_demo, extra_policy_update=extra_policy_update,
                           end_exploration_t=end_exploration_t, save_transitions=save_transitions)

    # Train DDPG agent with demonstrator transitions
    if ddpg:
        all_envs['DDPG'], all_losses['DDPG'], all_rewards['DDPG'], all_eval_results['DDPG'], \
        all_agents['DDPG'], all_envs['DDPG Best'] = \
            train_ddpg(schema=schema, episodes=episodes, random_seed=random_seed, batch_size=batch_size,
                       discount=discount, end_exploration_t=end_exploration_t, l2_loss=l2_loss)

    # Train SAC agent with demonstrator transitions
    if demo_transitions is not None and ddpg:
        all_envs['PRB_DDPG'], all_losses['PRB_DDPG'], all_rewards['PRB_DDPG'], all_eval_results['PRB_DDPG'], \
        all_agents['PRB_DDPG'] = \
            train_prbddpg(schema=schema, episodes=episodes, random_seed=random_seed, batch_size=batch_size,
                          discount=discount, demo_transitions=demo_transitions,
                          end_exploration_t=end_exploration_t, l2_loss=l2_loss)

    if demo_transitions is not None and include_sac:
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


def train_sacdemopol(schema, episodes, random_seed, batch_size, discount, autotune_entropy, clip_gradient,
                     kaiming_initialization, l2_loss, mode, imitation_lr, pretrained_demonstrator, deterministic_demo,
                     end_exploration_t, save_transitions):
    if pretrained_demonstrator is not None:
        with open(pretrained_demonstrator, 'rb') as file:
            pretrained_demonstrator = pickle.load(file)

    env = CityLearnEnv(schema)
    sacdemopol_model = SACDEMOPOL(env=env, seed=random_seed, batch_size=batch_size, autotune_entropy=autotune_entropy,
                                  clip_gradient=clip_gradient, kaiming_initialization=kaiming_initialization,
                                  l2_loss=l2_loss,
                                  discount=discount, mode=mode, imitation_lr=imitation_lr,
                                  pretrained_demonstrator=pretrained_demonstrator,
                                  deterministic_demo=deterministic_demo,
                                  end_exploration_time_step=end_exploration_t)
    losses, rewards, eval_results, best_state = sacdemopol_model.learn(episodes=episodes, deterministic_finish=True)

    best_state_env = get_best_env(sacdemopol_model, best_state)

    print('SAC-DemoPol model trained!')

    if save_transitions:
        save_transitions_to(env, sacdemopol_model, 'SAC-DemoPol')

    return env, losses, rewards, eval_results, sacdemopol_model, best_state_env


def train_sacdemoq(schema, episodes, random_seed, batch_size, discount, autotune_entropy, clip_gradient,
                   kaiming_initialization, l2_loss, imitation_lr, pretrained_demonstrator, deterministic_demo,
                   extra_policy_update, end_exploration_t, save_transitions):
    if pretrained_demonstrator is not None:
        with open(pretrained_demonstrator, 'rb') as file:
            pretrained_demonstrator = pickle.load(file)

    env = CityLearnEnv(schema)
    sacdemoq_model = SACDEMOQ(env=env, seed=random_seed, batch_size=batch_size, autotune_entropy=autotune_entropy,
                              clip_gradient=clip_gradient, kaiming_initialization=kaiming_initialization,
                              l2_loss=l2_loss,
                              discount=discount, imitation_lr=imitation_lr,
                              pretrained_demonstrator=pretrained_demonstrator,
                              deterministic_demo=deterministic_demo, extra_policy_update=extra_policy_update,
                              end_exploration_time_step=end_exploration_t,
                              n_interchanged_obs=len(env.interchanged_observations) * (len(env.buildings) - 1))
    losses, rewards, eval_results, best_state = sacdemoq_model.learn(episodes=episodes, deterministic_finish=True)

    best_state_env = get_best_env(sacdemoq_model, best_state)

    print('SAC-DemoQ model trained!')

    if save_transitions:
        save_transitions_to(env, sacdemoq_model, 'SAC-DemoQ')

    return env, losses, rewards, eval_results, sacdemoq_model, best_state_env


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


def train_prbddpg(schema, episodes, random_seed, batch_size, discount, demo_transitions, end_exploration_t, l2_loss):
    env = CityLearnEnv(schema)

    with open(demo_transitions, 'rb') as file:
        demo_transitions = pickle.load(file)

    prbddpg_model = PRBDDPG(env=env, seed=random_seed, batch_size=batch_size, l2_loss=l2_loss,
                            discount=discount, demonstrator_transitions=demo_transitions,
                            end_exploration_time_step=end_exploration_t)
    losses, rewards, eval_results, best_state = prbddpg_model.learn(episodes=episodes, deterministic_finish=True)

    print('PRB DDPG model trained!')

    return env, losses, rewards, eval_results, prbddpg_model


def train_ddpg(schema, episodes, random_seed, batch_size, discount, end_exploration_t, l2_loss):
    env = CityLearnEnv(schema)

    ddpg_model = DDPG(env=env, seed=random_seed, batch_size=batch_size, l2_loss=l2_loss,
                      discount=discount, demonstrator_transitions=demo_transitions,
                      end_exploration_time_step=end_exploration_t)
    losses, rewards, eval_results, best_state = ddpg_model.learn(episodes=episodes, deterministic_finish=True)

    best_state_env = get_best_env(ddpg_model, best_state)

    print('DDPG model trained!')

    return env, losses, rewards, eval_results, ddpg_model, best_state_env


if __name__ == '__main__':
    st = time.time()

    opts = parseOptions_social()

    DATASET_NAME = opts.schema
    seed = opts.seed
    building_count = opts.buildings
    demonstrators_count = opts.demonstrators
    episodes = opts.episodes
    discount = opts.discount
    include_tql = opts.include_tql
    include_rbc = opts.include_rbc
    include_sac = opts.include_sac
    include_sacdemopol = opts.include_sacdemopol
    include_sacdemoq = opts.include_sacdemoq
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
    ddpg = opts.ddpg

    if pretrained_demonstrator is not None:
        demonstrators_count = 1
    else:
        demonstrators_count = opts.demonstrators

    print(DATASET_NAME)
    train(dataset_name=DATASET_NAME, random_seed=seed, building_count=building_count,
          demonstrators_count=demonstrators_count, episodes=episodes, discount=discount,
          active_observations=active_observations, batch_size=batch_size, autotune_entropy=autotune_entropy,
          clip_gradient=clip_gradient, kaiming_initialization=kaiming_initialization, l2_loss=l2_loss,
          include_tql=include_tql, include_rbc=include_rbc, include_sac=include_sac,
          include_sacdemopol=include_sacdemopol,
          include_sacdemoq=include_sacdemoq,
          mode=mode, imitation_lr=imitation_lr, building_ids=building_ids, store_agents=store_agents,
          pretrained_demonstrator=pretrained_demonstrator, demo_transitions=demo_transitions,
          deterministic_demo=deterministic_demo, extra_policy_update=extra_policy_update,
          end_exploration_t=end_exploration_t, save_transitions=save_transitions, ddpg=ddpg)

    # get the end time
    et = time.time()

    # print the execution time
    elapsed_time = et - st
    print("")
    print('Execution time:', round(elapsed_time / 60), 'minutes')
