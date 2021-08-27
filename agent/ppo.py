from utils.utils import cal_discount_culmulative_reward, cal_gae
import torch
from agent.base_agent import BaseAgent
from memory.rollout_buffer import RolloutBuffer
from network.policies import ContinuousTDActor, DiscreteTDActor
from network.value_network import StateValueNetwork
import numpy as np

class PPOAgent(BaseAgent):
    def __init__(self, clipping_factor, discount_factor, entropy_factor,
            env, device) -> None:
        super().__init__(env=env, device=device)

        self.discount_factor = discount_factor
        self.clipping_factor = clipping_factor
        self.entropy_factor = entropy_factor

    def create_memory(self, memory_size=10000): 
        # if discrete action then shape is (1,)
        variable_dict={
            'state': self.state_shape,
            'action': (1,),
            'logprob': (1,),
            'reward': (1,),
            'done': (1,)
        }
        self.memory = RolloutBuffer(
            buffer_size=memory_size,
            variable_dict=variable_dict
        )
    
    def create_network(self, lr=(1e-4,1e-4)):
        super().create_network()
        if self.action_type == 'discrete':
            self.actor = DiscreteTDActor(
                lr=lr[0],
                state_shape=self.state_shape,
                n_actions=self.action_shape[0],
                device=self.device,
                n_hiddens=2,
                hidden_size=128,
                name='actor'
            )
        else:
            self.actor = ContinuousTDActor(
                lr=lr[0],
                state_shape=self.state_shape,
                action_shape=self.action_shape,
                device=self.device,
                n_hiddens=2,
                hidden_size=128,
                reshape_output=True,
                name='actor'
            )
        self.critic = StateValueNetwork(
            lr=lr[1],
            state_shape=self.state_shape,
            device=self.device,
            n_hiddens=2,
            hidden_size=128,
            name='critic'
        )
        self.network.append(self.actor)
        self.network.append(self.critic)
    
    def generate_trajectory(self, n_steps=1000, reset=True):
        if reset:
            state = self.env.reset()
        sum_reward = 0
        f = open('reward.txt','a')
        for _ in range(n_steps):
            state_tensor = torch.tensor(state).unsqueeze(0).float().to(self.device)
            action_tensor, logprob_tensor = self.actor.act(state_tensor)
            action = action_tensor.cpu().detach().numpy()
            logprob = logprob_tensor.cpu().detach().numpy()
            if self.action_type == 'discrete':
                next_state, reward, done, info = self.env.step(action[0])
            else:
                next_state, reward, done, info = self.env.step(action)
            self.env.render()
            sum_reward += reward
            reward = np.array([reward])
            done = np.array([done])
            
            self.memory.store(state, action, logprob, reward, done)
            if done[0]:
                state = self.env.reset()
                # print(sum_reward)
                f.write(str(sum_reward) + '\n')
                sum_reward = 0
            else:
                state = next_state
        f.close()

    
    def update_network_from_trajectory(self, n_steps, n_epochs):
        batch = self.memory.sample(sample_size=n_steps)
        state_tensor = torch.tensor(batch['state']).float().to(self.device)
        action_tensor = torch.tensor(batch['action']).float().to(self.device)
        old_logprob_tensor = torch.tensor(batch['logprob']).float().to(self.device)
        return_tensor = torch.tensor(
            cal_discount_culmulative_reward(
                batch['reward'], batch['done'], self.discount_factor)
        ).float().to(self.device)
        for _ in range(n_epochs):
            new_logprob_tensor, entropy_tensor = \
                self.actor.evaluate(state_tensor, action_tensor)
            value_tensor = self.critic.forward(state_tensor)

            advantage_tensor = return_tensor - value_tensor
            # next_state = torch.tensor(batch['state'][-1]).unsqueeze(0).float().to(self.device)
            # next_value = self.critic(next_state).cpu().detach().squeeze(0).numpy()[0]
            # advantage_tensor = torch.tensor(cal_gae(
            #     reward=batch['reward'],
            #     done=batch['done'],
            #     value=value_tensor.cpu().detach().numpy(),
            #     next_value=next_value,
            #     gamma=self.discount_factor,
            #     gae_lambda=0.95
            # )).float().to(self.device)

            ratio = torch.exp(new_logprob_tensor-old_logprob_tensor)
            surr1 = ratio * advantage_tensor
            surr2 = torch.clamp(ratio, 1.-self.clipping_factor, \
                1.+self.clipping_factor) * advantage_tensor
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * torch.square(value_tensor-return_tensor).mean()
            entropy_loss = -self.entropy_factor * entropy_tensor.mean()
            loss = actor_loss + critic_loss + entropy_loss
            # print(loss.item())
            loss_out = loss.item()

            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            loss.backward()
            self.actor.optimizer.step()
            self.critic.optimizer.step()
        print(loss_out)
    
    def learn(self, n_episodes, n_steps_per_episode):
        for episode in range(n_episodes):
            # self.load_network()
            self.generate_trajectory(n_steps_per_episode)
            self.update_network_from_trajectory(n_steps_per_episode, 30)
            # self.save_network()


        

# if __name__=='__main__':
#     ppo = PPOAgent(1,2)
#     print(ppo.network['test'])
    # for net in ppo.network:
    #     print(net)
        