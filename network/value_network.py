import torch
import torch.nn as nn
import torch.optim as optim
from network.base_network import BaseNetwork
from utils.utils import prod



#----------------------------------------------------------------------------
# network
#----------------------------------------------------------------------------
def prod(_tuple):
    product = 1
    for element in list(_tuple):
        product *= element
    return int(product)




class ActionValueNetwork(BaseNetwork):
    def __init__(self, lr, state_shape, action_shape, device, n_hiddens=2,
            hidden_size=256, name='critic_q', chkpt='./tmp') -> None:
        super().__init__(device, name, chkpt)
        self.state_shape = state_shape
        self.action_shape = action_shape

        # Input layer take flatened transform of input tensor
        self.input = nn.Sequential(
            nn.Linear(prod(self.state_shape) + prod(self.action_shape), hidden_size),
            nn.Tanh()
        )
        hidden_list = []
        for _ in range(n_hiddens - 1):
            hidden_list.append(nn.Linear(hidden_size, hidden_size))
            hidden_list.append(nn.Tanh())
        self.hidden = nn.Sequential(*hidden_list)
        self.output = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )

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
        self.state_shape = state_shape

        # Input layer take flatened transform of input tensor
        self.input = nn.Sequential(
            nn.Linear(prod(self.state_shape), hidden_size),
            nn.Tanh()
        )
        hidden_list = []
        for _ in range(n_hiddens - 1):
            hidden_list.append(nn.Linear(hidden_size, hidden_size))
            hidden_list.append(nn.Tanh())
        self.hidden = nn.Sequential(*hidden_list)
        self.output = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )

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