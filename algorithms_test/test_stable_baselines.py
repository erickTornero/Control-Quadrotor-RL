import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env =   gym.make('CartPole-v1')
env =   DummyVecEnv([lambda : env])

model   =   PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
cum_rw = 0
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    cum_rw += rewards
    env.render()

print(cum_rw)
env.close()