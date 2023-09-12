import pickle

from citylearn.agents.sac import SAC

from citylearn.data import DataSet

from citylearn.citylearn import CityLearnEnv

if __name__ == '__main__':
    env = CityLearnEnv(DataSet.get_schema('nydata'))
    agent = SAC(env=env)

    #env_history = ()

    with (open('../experiments/SAC_DB2/rewards_20230912T073504.pkl', 'rb')) as f:
        rewards = pickle.load(f)

    #envs = agent.get_env_history(directory='/Users/kristina/Documents/Studium/Informatik M.Sc. - Tü/SoSe 2023/social-rl/src')