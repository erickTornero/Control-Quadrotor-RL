import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO1

env = gym.make('CartPole-v0')
env = DummyVecEnv([lambda: env])

model = PPO1(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo1_cartpole")

del model # remove to demonstrate saving and loading

model = PPO1.load("ppo1_cartpole")

obs = env.reset()
c_rw = 0
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    c_rw += rewards
    env.render()
    if dones:
        print(c_rw)
        c_rw = 0

