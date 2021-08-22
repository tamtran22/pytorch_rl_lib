import torch
from agent.base_agent import BaseAgent
from memory.rollout_buffer import RolloutBuffer
from network.policies import DiscreteTDActor
from network.value_network import StateValueNetwork
import numpy as np

class PPOAgent(BaseAgent):
    def __init__(self, env, device) -> None:
        super().__init__(env=env, device=device)

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
    
    def create_network(self, lr=(1e-3,1e-3)):
        super().create_network()
        if self.action_type == 'discrete':
            self.network['actor'] = DiscreteTDActor(
                lr=lr[0],
                state_shape=self.state_shape,
                n_actions=self.action_shape[0],
                device=self.device,
                n_hiddens=2,
                hidden_size=128,
                name='actor'
            )
        self.network['critic'] = StateValueNetwork(
            lr=lr[1],
            state_shape=self.state_shape,
            device=self.device,
            n_hiddens=2,
            hidden_size=128,
            name='critic'
        )
    
    def generate_trajectory(self, n_episodes=1000, reset=True):
        if reset:
            state = self.env.reset()
        for _ in range(n_episodes):
            state_tensor = torch.tensor(state).unsqueeze(0).float().to(self.device)
            action_tensor, logprob_tensor = self.network['actor'].act(state_tensor)
            action = action_tensor.cpu().detach().squeeze(0).numpy()
            logprob = logprob_tensor.cpu().detach().numpy()
            next_state, reward, done, info = self.env.step(action)
            self.env.render()
            action = np.array([action])
            reward = np.array([reward])
            done = np.array([done])
            
            self.memory.store(state, action, logprob, reward, done)
            if done[0]:
                state = self.env.reset()
            else:
                state = next_state
        

# if __name__=='__main__':
#     ppo = PPOAgent(1,2)
#     print(ppo.network['test'])
    # for net in ppo.network:
    #     print(net)
        