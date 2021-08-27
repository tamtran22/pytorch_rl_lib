from agent.ppo import PPOAgent
import torch
import gym

if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = gym.make('MountainCarContinuous-v0')
    # env = gym.make('CartPole-v0')
    ppo = PPOAgent(
        clipping_factor=0.2,
        discount_factor=0.9,
        entropy_factor=0.1,
        env=env, 
        device=device
    )
    f = open('reward.txt','w')
    f.close()
    ppo.learn(100, 200)