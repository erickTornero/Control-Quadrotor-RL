import os
import gym
from tensorforce.agents import Agent
from tensorforce.execution import Runner
import sys
import json
import numpy as np
sys.path.insert(1, '../wrapper')
from wrapper_vrep import VREPQuad

env = VREPQuad(ip='127.0.0.1', port=19999)

restore_path = './models'
with open('./ppo_config.json', 'r') as fp:
    agent_conf = json.load(fp=fp)

network_spec = [
    dict(type='dense', size=64, activation='tanh'),
    dict(type='dense', size=64, activation='tanh')
]

agent = Agent.from_spec(
    spec=agent_conf,
    kwargs=dict(
        states=env.states,
        actions=env.actions,
        network=network_spec
    )
)

if os.path.exists(restore_path):
    print('Restoring model ...')
    agent.restore_model(directory=restore_path)


max_tsteps  = 250
n_episodes  =   100
for ep in range(1, n_episodes + 1):
    state = env.reset()
    cumrw = 0.0
    for _ in range(0, max_tsteps):
        actions=agent.act(state, deterministic=True)
        state, _, rw = env.execute(actions)
        cumrw   =   cumrw + rw
    
    vecpos = env.targetpos - state[9:12]
    vecpos = vecpos * vecpos
    print('{} episode\t-->\tEpisode Reward> {}\tFromTarget> {:6.4f}'.format(ep, cumrw, np.sqrt(vecpos.sum())))
    env.reset()