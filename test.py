from network.policies import DiscreteTDActor
from network.value_network import StateValueNetwork, ActionValueNetwork
from memory.rollout_buffer import RolloutBuffer
from agent.ppo import PPOAgent
import torch
import gym
# from network.base_network import BaseNetwork

if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # env = gym.make('MountainCarContinuous-v0')
    env = gym.make('CartPole-v0')
    ppo = PPOAgent(env, device)
    print(ppo.state_shape, ppo.action_shape)
    for net in ppo.network:
        print(ppo.network[net])