import gym

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

n_cpu = 2

env = SubprocVecEnv([lambda: gym.make('Enduro-v0') for i in range(n_cpu)])


#model = PPO2(CnnPolicy, env, verbose=1)
#model.learn(total_timesteps=25000)
#
#model.save('ppo2-Enduro-v0')
#
#del model

model = PPO2.load('ppo2-Enduro-v0')


for i in range(10):
    obs = env.reset()
    c_rw = 0
    done = False

    while True:
        action, _states = model.predict(obs)
        obs, rw, done, info = env.step(action)
        print(done)
        c_rw = c_rw + rw
        env.render()
    
    print(i+1, c_rw)
    c_rw = 0