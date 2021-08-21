from agent.base_agent import BaseAgent
from memory.rollout_buffer import RolloutBuffer
from network.policies import DiscreteTDActor
from network.value_network import StateValueNetwork

class PPOAgent(BaseAgent):
    def __init__(self, env, device) -> None:
        super().__init__(env=env, device=device)

    def create_memory(self, memory_size=10000): 
        variable_dict={
            'state': self.state_shape,
            'action': self.action_shape,
            'logprob': self.action_shape,
            'reward': (1,),
            'done': (1,)
        }
        self.memory = RolloutBuffer(
            buffer_size=memory_size,
            variable_dict=variable_dict
        )
    
    def create_network(self):
        super().create_network()
        self.network['actor'] = DiscreteTDActor(
            lr=1e-4,
            state_shape=self.state_shape,
            n_actions=self.action_shape[0],
            device=self.device,
            n_hiddens=2,
            hidden_size=128,
            name='actor'
        )
        self.network['critic'] = StateValueNetwork(
            lr=1e-4,
            state_shape=self.state_shape,
            device=self.device,
            n_hiddens=2,
            hidden_size=128,
            name='critic'
        )
if __name__=='__main__':
    ppo = PPOAgent(1,2)
    print(ppo.network['test'])
    # for net in ppo.network:
    #     print(net)
        