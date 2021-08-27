import gym as G 
import numpy as N

class DummyEnv(G.Env):
    metadata = {'render.modes':['human']}
    def __init__(self, d = 1, range = 10.):
        super(DummyEnv, self).__init__()
        self.d = d
        mean_state = self.reset()
        self.action_space = G.spaces.Box(
            low=-range,
            high=range,
            shape=(self.d,),
            dtype=N.float
        )
        self.observation_space = G.spaces.Box(
            low=mean_state-range,
            high=mean_state+range,
            shape=(self.d,),
            dtype=N.float
        )
    def step(self, action):
        next_state = self.state.copy()
        next_state += N.array(action)
        reward = - self.objective(next_state)
        f = open("reward","a+")
        f.write(str(reward)+"\n")
        f.close()
        done = self.is_done(next_state)
        info = {}
        self.reset()
        return self.state, reward, done, info

    def reset(self):
        self.state = N.zeros(shape=(self.d,))
        return self.state
    def render(self):
        raise NotImplementedError
    def objective(self, x):
        s = 0
        for i in range(len(x)):
            s += (x[i]-3)**2
        return s
    def is_done(self, state):
        checked = False
        for i in range(self.d):
            if (state[i] < self.observation_space.low[i]) or (state[i] > self.observation_space.high[i]):
                checked = True
                break
        return checked