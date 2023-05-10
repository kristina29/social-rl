from matplotlib import pyplot as plt

from citylearn.citylearn import CityLearnEnv

from citylearn.data import DataSet
from citylearn.agents.marlisa import MARLISA
from citylearn.reward_function import MARL
from utils import set_schema_buildings, set_schema_simulation_period, set_active_observations, plot_simulation_summary

DATASET_NAME = 'citylearn_challenge_2022_phase_all'
schema = DataSet.get_schema(DATASET_NAME)
root_directory = schema['root_directory']

RANDOM_SEED = 0
print('Random seed:', RANDOM_SEED)

# edit next code line to change number of buildings in simulation
building_count = 5

 # edit next code line to change number of days in simulation
day_count = 28

# edit next code line to change active observations in simulation
active_observations = ['hour', 'outdoor_dry_bulb_temperature', 'outdoor_dry_bulb_temperature_predicted_6h',
                       'direct_solar_irradiance', 'net_electricity_consumption']

schema, buildings = set_schema_buildings(schema, building_count, RANDOM_SEED)
schema, simulation_start_time_step, simulation_end_time_step = set_schema_simulation_period(schema, day_count,
                                                                                            RANDOM_SEED, root_directory)
schema, active_observations = set_active_observations(schema, active_observations)

print('Selected buildings:', buildings)
print(
    f'Selected {day_count}-day period time steps:',
    (simulation_start_time_step, simulation_end_time_step)
)
print(f'Active observations:', active_observations)

schema['central_agent'] = False
env = CityLearnEnv(schema)
env.reward_function = MARL(env)

marlisa_model = MARLISA(env, start_training_time_step=50, information_sharing=True)

# ----------------- CALCULATE NUMBER OF TRAINING EPISODES -----------------
marlisa_episodes = 2
print('Number of episodes to train:', marlisa_episodes)

# ------------------------------- TRAIN MODEL -----------------------------
marlisa_model.learn(episodes=marlisa_episodes, deterministic_finish=True)

observations = env.reset()
marlisa_actions_list = []

while not env.done:
    actions = marlisa_model.predict(observations, deterministic=True)
    observations, _, _, _ = env.step(actions)
    marlisa_actions_list.append(actions)

plot_simulation_summary(
    {'MARLISA': env}
)

#fig = plot_actions(sacr_actions_list, 'SAC Actions using Custom Reward')
plt.show()

#dataset_name = 'citylearn_challenge_2022_phase_1'
#env = CityLearnEnv(dataset_name)
#model = MARLISA(env)
#model.learn(episodes=2, deterministic_finish=True)