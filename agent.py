from abc import ABC
from math import e
from typing import TextIO

from memory import RolloutBuffer
from network import TDActorNetwork, StateValueNetwork
import torch
import gym
import numpy as np



#------------------------------------------------------------------------
# base agent class
#------------------------------------------------------------------------
class BaseAgent(ABC):
    def __init__(self) -> None:
        super().__init__()
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
        super().__init__()
        if device!=None:
            self.device = device
        else:
            self.device = torch.device(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        if env != None:
            self.get_env(env)
            self.get_memory()
            self.get_network(lr)

    def get_env(self, env):
        self.env = env
        # check discrete and continious space
        if self.env.observation_space.shape == ():
            # discrete space
            # self.state_shape = (self.env.observation_space.n,)
            self.state_shape = (1,)
        else:
            # continuous space
            self.state_shape = self.env.observation_space.shape
        if env.action_space.shape == ():
            # self.action_shape = (self.env.action_space.n,)
            self.action_shape = (1,)
        else:
            self.action_shape = self.env.action_space.shape
    def get_memory(self):
        variable_dict = {
            'state': self.state_shape,
            'action': self.action_shape,
            'logprob': self.action_shape,
            'reward': (1,),
            'value': (1,),
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
    def store(self, state, action, logprob, reward, value, done):
        self.memory.store(state, action, logprob, reward, value, done)
    def generate_trajectory(self, n_steps, reset=True):
        if reset:
            state = self.env.reset()
        for _ in range(n_steps):
            state_tensor = torch.tensor(state).unsqueeze(0).float().to(self.device)
            action_tensor, logprob_tensor = self.actor.act(state_tensor)
            action = action_tensor.cpu().detach().squeeze(0).numpy()
            logprob = logprob_tensor.cpu().detach().squeeze(0).numpy()
            next_state, reward, _done, info = self.env.step(action)
            reward = np.array([reward])
            done = np.array([_done])
            value_tensor = self.critic.forward(state_tensor)
            value = value_tensor.cpu().detach().squeeze(0).numpy()
            # print(state, action, logprob, reward, value, done)
            self.store(state, action, logprob, reward, value, done)
            if _done:
                state = self.env.reset()
            else:
                state = next_state
        
    def learn_trajectory(self, batch_size):
        batch = self.memory.sample(sample_size=batch_size)
        state_tensor = torch.tensor(batch['state']).float().to(self.device)
        action_tensor = torch.tensor(batch['action']).float().to(self.device)
        old_logprob_tensor = torch.tensor(batch['logprob']).float().to(self.device)
        new_logprob_tensor = self.actor.evaluate(state_tensor, action_tensor)
        # print(state_tensor.shape, action_tensor.shape, old_logprob_tensor.shape, new_logprob_tensor.shape)
        print(batch['done'])
        # for i in range(batch_size):







#-------------------------------------------------------------------------
# main function
#-------------------------------------------------------------------------
if __name__ == '__main__':
    env = gym.make("MountainCarContinuous-v0")
    agent = PPOAgent(
        lr=(1e-3, 1e-3),
        env=env,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    agent.generate_trajectory(
        n_steps=10,
        reset=True
    )
    agent.learn_trajectory(4)
    # print(agent.memory.variable)
    # print(agent.state_shape)
    # print(agent.action_shape)



