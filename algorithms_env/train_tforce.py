import sys

sys.path.insert(1, '../wrapper')
from wrapper_vrep import VREPQuad

import gym
import tensorflow as tf
import json

from tensorforce.agents import Agent
from tensorforce.execution import Runner

from tensorboardX import SummaryWriter

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)


network_spec = [
    dict(type='dense', size=64, activation='tanh'),
    dict(type='dense', size=64, activation='tanh')]

env =   VREPQuad(ip='127.0.0.1', port=19999)
with open('./ppo_config.json', 'r') as fp:
    agent_conf = json.load(fp=fp)

agent_conf['execution']['session_config'] = tf.ConfigProto(gpu_options=gpu_options)


agent = Agent.from_spec(
    spec=agent_conf,
    kwargs=dict(
        states=env.states,
        actions=env.actions,
        network=network_spec
    )
)
import os
import numpy as np
save_model_path = "./models/model"
backup_path = "./stamps/model"
restore_path = "./models"
if os.path.exists(restore_path):
    print('Restore model ...')
    agent.restore_model(directory=restore_path)

writer = SummaryWriter()

def callback(info):
    writer.add_scalar('data/rewards', info.episode_rewards[-1], info.episode)
    if info.episode % 10 == 0:
        print('Episode:>{}\ttimesteps> {}\t Ep Reward> {:05.2f}'.format(info.episode,info.timestep, np.mean(info.episode_rewards[-10:])))
    if info.episode % 100 == 0:
        print('Saving model ...')
        agent.save_model(directory=save_model_path, append_timestep=False)
    
    return True


runner = Runner(agent, env)

runner.run(num_timesteps=7000000, num_episodes=500000, max_episode_timesteps=250, episode_finished=callback)
runner.close()
writer.close()