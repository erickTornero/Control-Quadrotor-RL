#from wrapper.wrapper_vrep import *
#from wrapper.utility import *
import sys
sys.path.insert(1, '../wrapper')
from wrapper_vrep import VREPQuad
print(sys.path)
import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv

from stable_baselines import PPO2


n_cpu   =   2
env     =   SubprocVecEnv([lambda: VREPQuad(ip='192.168.0.36', port=19999) for i in range(1)])
print(env)

model   =   PPO2(MlpPolicy, env, verbose=1, learning_rate=1e-3)
model.learn(total_timesteps=100)
model.save('ppo2_quad')

obs =   env.reset()

while True:
    act, _states = model.predict(obs)
    obs, rw, dones, info = env.step(act)
    #env.render()