import sys

sys.path.insert(1, '../wrapper')
from wrapper_vrep import VREPQuad

import gym
import tensorflow as tf
import json

from tensorforce.agents import Agent
from tensorforce.execution import Runner


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

runner = Runner(agent, env)

runner.run(num_timesteps=7000000, num_episodes=500000, max_episode_timesteps=250)
runner.close()