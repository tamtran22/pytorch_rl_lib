import gym
from gym import spaces
import numpy as np
from numpy.lib.utils import info
import math

class FunctionEnv(gym.Env):
    """Real function on R^d"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(FunctionEnv, self).__init__()
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float
        )
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float
        )
        self.reset()
    
    def step(self, action):
        self.state[0] += action[0]
        self.state[1] += action[1]
        observation = self.state
        reward = self.function()
        done = self.is_done()
        info = {}
        # print(observation, reward)
        self.reset()
        return observation, reward, done, info
    
    def reset(self):
        self.state = np.zeros(self.observation_space.shape)
        return self.state
    def render(self):
        raise NotImplementedError
    def function(self):
        return -math.sqrt((self.state[0]-0.3)**2 + (self.state[1]+0.5)**2)
    def is_done(self):
        out_of_x_range = (self.state[0] < -1) \
            or (self.state[0] > 1)
        out_of_y_range = (self.state[1] < -1) \
            or (self.state[1] > 1)
        return out_of_x_range or out_of_y_range