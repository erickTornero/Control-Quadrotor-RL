# This will envolve the environment VREP quadrotor by an openai gym

import gym
from gym import spaces
import numpy as np

from utility import GetFlatRotationMatrix
# From environment
import vrep
from typing import NoReturn
import time

# Environment to pass the target position just on the initial state
class VREPQuad(gym.Env):

    def __init__(self, ip='127.0.0.1', port=19997, envname='Quadricopter', targetpos=np.zeros(3, dtype=np.float32)):
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
        
        pass

        if not self._get_boolparam(vrep.sim_boolparam_headless):
            self._clear_gui()

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
        #r = vrep.simxSetObjectPosition(self.clientID, self.quad_handler, -1, np.array([0.0,0.0,0.5]), vrep.simx_opmode_oneshot_wait)
        #r = vrep.simxCallScriptFunction(self.clientID, 'Quadricopter_target', vrep.sim_scripttype_childscript, 'sysCall_custom_reset', np.array([]), np.array([]), np.array([]), bytearray(), vrep.simx_opmode_blocking)
        # pass
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)
        ## while True:
        ##     e = vrep.simxGetInMessageInfo(self.clientID, vrep.simx_headeroffset_server_state)
        ##     still_running = e[1] & 1
        ##     print(e)
        ##     if not still_running:
        ##         break
        ##
        time.sleep(3.0)
 
        # Reset quadrotor
        r, self.quad_handler         =   vrep.simxGetObjectHandle(self.clientID, self.envname, vrep.simx_opmode_oneshot_wait)
        # start pose

        # Start simulation
        print('Starting simulation')
        #vrep.simxSynchronousTrigger(self.clientID)
        #vrep.simxGetPingTime(self.clientID)
        self.startsimulation()
        #vrep.simxSynchronous(self.clientID, True)
        #e = vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
        #print(e)
        # get observation


        vrep.simxSynchronousTrigger(self.clientID)
        vrep.simxGetPingTime(self.clientID)
        #vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
#
        #vrep.simxSynchronous(self.clientID, True)
        #vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
        #print('s')

    def render(self, close=False):
        # Put code if it is necessary to render
        pass

    def startsimulation(self):
        if self.clientID != -1:
            self._set_floatparam(vrep.sim_floatparam_simulation_time_step, 0.05)
            vrep.simxSynchronous(self.clientID, True)
            e = vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)

            self._set_boolparam(vrep.sim_boolparam_threaded_rendering_enabled, True)
            print(e)
        else:
            raise ConnectionError('Any conection has been done')
    def __del__(self):
        print('Exit connection')
        vrep.simxFinish(-1)

    def _set_floatparam(self, parameter: int, value: float) ->NoReturn:
        res =   vrep.simxSetFloatingParameter(self.clientID, parameter, value, vrep.simx_opmode_oneshot)
        print(res)
        assert (res == vrep.simx_return_ok or res == vrep.simx_return_novalue_flag), ('Could not set float parameters!')

    def _set_boolparam(self, parameter: int, value: bool) -> NoReturn:
        """Sets boolean parameter of V-REP simulation.
        Args:
            parameter: Parameter to be set.
            value: Boolean value to be set.
        """
        res = vrep.simxSetBooleanParameter(self.clientID, parameter, value,
                                           vrep.simx_opmode_oneshot)
        assert (res == vrep.simx_return_ok or res == vrep.simx_return_novalue_flag), (
            'Could not set boolean parameter!')

    def _clear_gui(self) -> NoReturn:
        """Clears GUI with unnecessary elements like model hierarchy, library browser and
        console. Also this method enables threaded rendering.
        """
        self._set_boolparam(vrep.sim_boolparam_hierarchy_visible, False)
        self._set_boolparam(vrep.sim_boolparam_console_visible, False)
        self._set_boolparam(vrep.sim_boolparam_browser_visible, False)

    def _get_boolparam(self, parameter: int) -> bool:
        res, value = vrep.simxGetBooleanParameter(self.clientID, parameter,
                                                  vrep.simx_opmode_oneshot)
        assert (res == vrep.simx_return_ok or res == vrep.simx_return_novalue_flag), (
            'Could not get boolean parameter!')
        return value

## Test

vrepX = VREPQuad(ip='192.168.0.36',port=19999)

import time
time.sleep(1)

ob, rw =    vrepX.step(np.array([4.4, 4, 4.3, 4])) 
print('observation> ', ob)
print('reward>', rw)
