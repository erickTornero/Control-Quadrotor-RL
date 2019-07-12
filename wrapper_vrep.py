# This will envolve the environment VREP quadrotor by an openai gym

import gym
from gym import spaces
import numpy as np

# From environment
import vrep

class VREPQuad(gym.Env):

    def __init__(self, ip='127.0.0.1', port='5000', envname='Quad_p3dx'):
        super(VREPQuad, self).__init__()
        # Initialize vrep
        self.envname            =   envname
        vrep.simxFinish(-1)
        clientID                =   vrep.simxStart(ip, port, True, True, 5000, 5)
        if clientID != -1:
            print('Connection Established Successfully')
        else:
            raise ConnectionError("Can't Connect with the envinronment at IP:{}, Port:{}".format(ip, port))
        
        _, quad_handler         =   vrep.simxGetObjectHandle(clientID, self.envname, vrep.simx_opmode_oneshot_wait)

        # Define gym variables
        
        self.action_space       =   spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        self.observation_space  =   spaces.Box(low=-1000.0, high=1000.0, shape=(18,), dtype=np.float32)

    def step(self, action):
        # Put code here:
        pass

    def reset(self):
        # Put code when reset here
        pass

    def render(self, close=False):
        # Put code if it is necessary to render
        pass


