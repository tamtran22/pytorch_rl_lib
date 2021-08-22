import torch
import torch.nn as nn
import torch.optim as optim
from network.base_network import BaseNetwork
from utils.utils import prod



#----------------------------------------------------------------------------
# network
#----------------------------------------------------------------------------

class ActionValueNetwork(BaseNetwork):
    def __init__(self, lr, state_shape, action_shape, device, n_hiddens=2,
            hidden_size=256, name='critic_q', chkpt='./tmp') -> None:
        super().__init__(device, name, chkpt)
        self.input_shape = (prod(state_shape)+prod(action_shape),)
        self.output_shape = (1,)


        # Input layer take flatened transform of input tensor
        self.create_input_layer(self.input_shape, hidden_size, nn.ReLU())
        self.create_hidden_layer(n_hiddens, hidden_size, nn.ReLU())
        self.create_output_layer(hidden_size, self.output_shape, nn.ReLU())

        self.optimizer = optim.Adam(
            self.parameters(), 
            lr=lr
        )
        self.to(self.device)
    
    def forward(self, state, action):
        x = torch.cat([
            torch.flatten(state, start_dim=1), 
            torch.flatten(action, start_dim=1)
        ], dim=1).to(self.device)
        x = self.input(x)
        x = self.hidden(x)
        x = self.output(x)
        return x




class StateValueNetwork(BaseNetwork):
    def __init__(self, lr, state_shape, device, n_hiddens=2,
            hidden_size=256, name='critic_v', chkpt='./tmp') -> None:
        super().__init__(device, name, chkpt)
        self.input_shape = state_shape
        self.output_shape = (1,)

        # Input layer take flatened transform of input tensor
        self.create_input_layer(self.input_shape, hidden_size, nn.SiLU())
        self.create_hidden_layer(n_hiddens, hidden_size, nn.SiLU())
        self.create_output_layer(hidden_size, self.output_shape, nn.SiLU())

        self.optimizer = optim.Adam(
            self.parameters(), 
            lr=lr
        )
        self.to(self.device)
    
    def forward(self, state):
        x = torch.flatten(state, start_dim=1).to(self.device)
        x = self.input(x)
        x = self.hidden(x)
        x = self.output(x)
        return x