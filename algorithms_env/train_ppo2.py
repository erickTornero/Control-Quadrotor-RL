#from wrapper.wrapper_vrep import *
#from wrapper.utility import *
import sys
sys.path.insert(1, '../wrapper')
from wrapper_vrep import VREPQuad
print(sys.path)
import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv

import numpy as np
n_stp = 0

from stable_baselines import PPO2
def callback(_locals, _globals):
    global n_stp
    if (n_stp + 1) % 10 == 0:
        _locals['self'].save('ppo2_quad_fourth')
    n_stp = n_stp + 1
    return True

n_cpu   =   2
env     =   SubprocVecEnv([lambda: VREPQuad(ip='127.0.0.1', port=19999) for i in range(1)])
print(env)

model   =   PPO2(MlpPolicy, env, verbose=1, learning_rate=1e-3, n_steps=1024, tensorboard_log='runs')
model.learn(total_timesteps=7500000, callback=callback)
model.save('ppo2_quad')
## Get the mean
# model.policy_proba[0] to get the mean, since it's property of policy father actor critic
#if TRAINING==True:
#model = PPO2.load('ppo2_quad_third', env=env)

obs =   env.reset() 

while True:
    #print('===================================')
    act, _states = model.predict(obs)
    #print(act)
    #act, _states = model.predict(obs)
    #print(act)
    #act, _states = model.predict(obs)
    #print(act)
    #act, _states = model.predict(obs)
    #print(act)
    #print(model.action_probability(obs)[0].squeeze(0))
    #act = model.action_probability(obs)[0]
    #print(act)
    #print('===================================')
    
    obs, rw, dones, info = env.step(act)
    #env.render()
