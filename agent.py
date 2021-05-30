from abc import ABC
from math import e
from typing import TextIO

from torch._C import LoggerBase
from memory import RolloutBuffer
from network import TDActorNetwork, StateValueNetwork
import torch
import gym
import numpy as np



#------------------------------------------------------------------------
# base agent class
#------------------------------------------------------------------------
class BaseAgent(ABC):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device
    def get_env(self):
        pass
    def get_memory(self):
        pass
    def get_network(self):
        pass
    def record(self):
        pass
    def save_model(self):
        pass
    def update(self):
        pass
    def learn(self):
        pass




#------------------------------------------------------------------------
# PPO agent
#------------------------------------------------------------------------
class PPOAgent(BaseAgent):
    def __init__(self, lr,  env=None, device=None) -> None:
        super().__init__(device)
        if env != None:
            self.get_env(env)
            self.get_memory()
            self.get_network(lr)


    def get_env(self, env):
        self.env = env
        # check discrete and continious space
        if self.env.observation_space.shape == ():
            # discrete space
            self.state_shape = (self.env.observation_space.n,)
        else:
            # continuous space
            self.state_shape = self.env.observation_space.shape
        if env.action_space.shape == ():
            self.action_shape = (self.env.action_space.n,)
        else:
            self.action_shape = self.env.action_space.shape
    def get_memory(self):
        variable_dict = {
            'state': self.state_shape,
            'action': self.action_shape,
            'logprob': self.action_shape,
            'reward': (1,),
            'state_value': (1,),
            'done': (1,)
        }
        self.memory = RolloutBuffer(
            buffer_size=100,
            variable_dict=variable_dict
        )
    def get_network(self,lr=(None, None)):
        self.actor = TDActorNetwork(
            lr=lr[0],
            state_shape=self.state_shape,
            action_shape=self.action_shape,
            device=self.device
        )
        self.critic = StateValueNetwork(
            lr=lr[1],
            state_shape=self.state_shape,
            device=self.device
        )
    def store(self, state, action, logprob, reward, state_value, done):
        self.memory.store([state, action, logprob, reward, state_value, done])
    def generate_trajectory(self, n_steps, reset=True):
        if reset:
            state = self.env.reset()
        for _ in range(n_steps):
            action, logprob = self.actor.act(torch.tensor(state).unsqueeze(0).to(self.device))
            action = action.squeeze(0).numpy()
            logprob = logprob.squeeze(0).numpy()
            next_state, reward, _done, info = self.env.step(action)
            reward = np.array([reward])
            done = np.array([_done])
            state_value = self.critic.forward(torch.tensor(state).unsqueeze(0).to(self.device))
            state_value = state_value.squeeze(0).numpy()
            self.store(state, action, logprob, reward, state_value, done)
            if _done:
                state = self.env.reset()
        
    def learn_trajectory(self, batch_size):
        batch = self.memory.sample(sample_size=batch_size)





#-------------------------------------------------------------------------
# main function
#-------------------------------------------------------------------------
if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    agent = PPOAgent(
        lr=(1e-3, 1e-3),
        env=env,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    print(type(env.reset()))
    # print(agent.memory.variable)
    # print(agent.state_shape)
    # print(agent.action_shape)



