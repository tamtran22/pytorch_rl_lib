from abc import ABC
from math import e
from typing import TextIO

from memory.rollout_buffer import RolloutBuffer
from network.network import TDActorNetwork, StateValueNetwork
from utils.utils import cal_discount_culmulative_reward
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
    def __init__(self, lr, clipping_factor, discount_factor, entropy_factor,
            env=None, device=None) -> None:
        super().__init__()
        if device!=None:
            self.device = device
        else:
            self.device = torch.device(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        if env != None:
            self.get_env(env)
            self.get_memory()
            self.get_network(lr)
        self.discount_factor = discount_factor
        self.clipping_factor = clipping_factor
        self.entropy_factor = entropy_factor

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
            'done': (1,)
        }
        self.memory = RolloutBuffer(
            buffer_size=10000,
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
    def store(self, state, action, logprob, reward, done):
        self.memory.store(state, action, logprob, reward, done)
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

            # print(state, action, logprob, reward, value, done)
            self.store(state, action, logprob, reward, done)
            if _done:
                state = self.env.reset()
            else:
                state = next_state
        print(self.memory.variable['reward'].mean())
        
    def learn_trajectory(self, batch_size, n_epochs):
        batch = self.memory.sample(sample_size=batch_size)

        state_tensor = torch.tensor(batch['state']).float().to(self.device)
        action_tensor = torch.tensor(batch['action']).float().to(self.device)
        old_logprob_tensor = torch.tensor(batch['logprob']).float().to(self.device)
        return_tensor = torch.tensor(
            cal_discount_culmulative_reward(batch['reward'], batch['done'], self.discount_factor)
        ).float().to(self.device)

        for _ in range(n_epochs):
            new_logprob_tensor, entropy_tensor = self.actor.evaluate(state_tensor, action_tensor)
            value_tensor = self.critic.forward(state_tensor)
            advantage_tensor = return_tensor - value_tensor
            ratio = torch.exp(new_logprob_tensor - old_logprob_tensor)
            surr1 = ratio * advantage_tensor
            surr2 = torch.clamp(ratio, 1.-self.clipping_factor, \
                1+self.clipping_factor) * advantage_tensor
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * torch.square(value_tensor-return_tensor).mean()
            entropy_loss = -self.entropy_factor * entropy_tensor.mean()
            loss = actor_loss + critic_loss

            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            loss.backward()
            # actor_loss.backward(retain_graph=True)
            # critic_loss.backward(retain_graph=True)
            # entropy_loss.backward()
            self.actor.optimizer.step()
            self.critic.optimizer.step()
        
        # self.save_model()
            

            

    def save_model(self):
        print("Saving networks...")
        self.actor.save()
        self.critic.save()
    def load_model(self):
        self.actor.load()
        self.critic.load()





#-------------------------------------------------------------------------
# main function
#-------------------------------------------------------------------------
if __name__ == '__main__':
    env = gym.make("MountainCarContinuous-v0")
    agent = PPOAgent(
        lr=(1e-3, 1e-3),
        clipping_factor=0.9,
        discount_factor=0.99,
        entropy_factor=0.1,
        env=env,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    for i in range(1000):
        # print("Iteration ", i)
        agent.generate_trajectory(
            n_steps=100,
            reset=True
        )
        agent.learn_trajectory(100, 50)
    # print(agent.memory.variable)
    # print(agent.state_shape)
    # print(agent.action_shape)



