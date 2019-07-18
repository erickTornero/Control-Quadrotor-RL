# This will envolve the environment VREP quadrotor by an openai gym

import gym
from gym import spaces
import numpy as np

from utility import GetFlatRotationMatrix
# From environment
import vrep

# Environment to pass the target position just on the initial state
class VREPQuad(gym.Env):

    def __init__(self, ip='127.0.0.1', port=19999, envname='Quadricopter_base', targetpos=np.zeros(3, dtype=np.float32)):
        super(VREPQuad, self).__init__()
        # Initialize vrep
        self.envname            =   envname
        vrep.simxFinish(-1)
        clientID                =   vrep.simxStart(ip, port, True, True, 5000, 5)
        if clientID != -1:
            print('Connection Established Successfully')
            self.clientID       =   clientID
            self.targetpos      =   targetpos
        else:
            raise ConnectionError("Can't Connect with the envinronment at IP:{}, Port:{}".format(ip, port))
        
        r, self.quad_handler         =   vrep.simxGetObjectHandle(clientID, self.envname, vrep.simx_opmode_oneshot_wait)

        print(r, self.quad_handler)
        # Define gym variables

        self.action_space       =   spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        self.observation_space  =   spaces.Box(low=-1000.0, high=1000.0, shape=(18,), dtype=np.float32)

        # Get scripts propellers Here...!
        self.propsignal =   ['speedprop' + str(i+1) for i in range(0, 4)]
        
        #self.propsignal2 = 'speedprop2'
        #self.propsignal3 = 'speedprop3'
        #self.propsignal4 = 'speedprop4'

    def step(self, action):
        # assume of action be an np.array of dimension (4,)
        # Act!
        for act, name in zip(action, self.propsignal):
            vrep.simxSetFloatSignal(self.clientID, name, act, vrep.simx_opmode_streaming)
        
        #vrep.simxSetFloatSignal(self.clientID, self.propsignal1)
        # Put code here:

        ## Do an action to environment

        ## Get states
        _, position        =   vrep.simxGetObjectPosition(self.clientID,    self.quad_handler, -1, vrep.simx_opmode_oneshot_wait)
        orientation     =   vrep.simxGetObjectOrientation(self.clientID, self.quad_handler, -1, vrep.simx_opmode_oneshot_wait)
        velocity        =   vrep.simxGetObjectVelocity(self.clientID,    self.quad_handler, vrep.simx_opmode_oneshot_wait)
        quaternion      =   vrep.simxGetObjectQuaternion(self.clientID,  self.quad_handler, -1, vrep.simx_opmode_oneshot_wait)

        # Flat and join states!
        RotMat          =   GetFlatRotationMatrix(orientation[1])
        rowdata         =   np.append(RotMat, position)
        rowdata         =   np.append(rowdata, velocity[1])
        rowdata         =   np.append(rowdata, velocity[2])

        reward  =   self.targetpos - position
        reward  =   20.0 - np.sqrt((reward * reward).sum()) 
        
        return (rowdata, reward)

        # Compute The reward function

    def reset(self):
        # Put code when reset here
        pass

    def render(self, close=False):
        # Put code if it is necessary to render
        pass
    def __del__(self):
        print('Exit connection')
        vrep.simxFinish(-1)


## Test

vrepX = VREPQuad(ip='192.168.0.36',port=19999)

import time
time.sleep(1)

ob, rw =    vrepX.step(np.array([4.4, 8, 4.3, 8])) 
print('observation> ', ob)
print('reward>', rw)
