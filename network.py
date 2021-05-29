from abc import ABC
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


#---------------------------------------------------------------------------
# Base network class
#---------------------------------------------------------------------------
class BaseNetwork(nn.Module, ABC):
    def __init__(self, device, name, chkpt) -> None:
        super().__init__()
        self.name = name
        self.checkpoint_file = os.path.join(chkpt, name)
        self.device = device
    def save(self):
        torch.save(self.state_dict(), self.checkpoint_file)
    def load(self):
        self.load_state_dict(torch.load(self.checkpoint_file))



#----------------------------------------------------------------------------
# Policies
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
        self.input = nn.Linear(prod(self.state_shape) + prod(self.action_shape), hidden_size)
        hidden_list = []
        for _ in range(n_hiddens - 1):
            hidden_list.append(nn.Linear(hidden_size, hidden_size))
            hidden_list.append(nn.ReLU(inplace=True))
        self.hidden = nn.Sequential(*hidden_list)
        self.output = nn.Linear(hidden_size, 1)

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
        x = F.relu(x)
        x = self.hidden(x)
        x = self.output(x)
        x = torch.sigmoid(x)
        return x




class StateValueNetwork(BaseNetwork):
    def __init__(self, lr, state_shape, device, n_hiddens=2,
            hidden_size=256, name='critic_v', chkpt='./tmp') -> None:
        super().__init__(device, name, chkpt)
        self.state_shape = state_shape

        # Input layer take flatened transform of input tensor
        self.input = nn.Linear(prod(self.state_shape), hidden_size)
        hidden_list = []
        for _ in range(n_hiddens - 1):
            hidden_list.append(nn.Linear(hidden_size, hidden_size))
            hidden_list.append(nn.ReLU(inplace=True))
        self.hidden = nn.Sequential(*hidden_list)
        self.output = nn.Linear(hidden_size, 1)

        self.optimizer = optim.Adam(
            self.parameters(), 
            lr=lr
        )
        self.to(self.device)
    
    def forward(self, state):
        x = torch.flatten(state, start_dim=1).to(self.device)
        x = self.input(x)
        x = F.relu(x)
        x = self.hidden(x)
        x = self.output(x)
        x = torch.sigmoid(x)
        return x




class GaussianPolicyNetwork(BaseNetwork):
    def __init__(self, lr, state_shape, action_shape, device, n_hiddens=2,
            hidden_size=256, name='actor_gauss', chkpt='./tmp') -> None:
        super().__init__(device, name, chkpt)
        self.state_shape = state_shape
        self.action_shape = action_shape

        # Flaten input tensor
        self.input = nn.Linear(prod(self.state_shape), hidden_size)
        hidden_list = []
        for _ in range(n_hiddens - 1):
            hidden_list.append(nn.Linear(hidden_size, hidden_size))
            hidden_list.append(nn.ReLU(inplace=True))
        self.hidden = nn.Sequential(*hidden_list)
        self.sigma = nn.Linear(hidden_size, prod(self.action_shape))
        self.mu = nn.Linear(hidden_size, prod(self.action_shape))

        self.optimizer = optim.Adam(
            self.parameters(), 
            lr=lr
        )
        self.to(self.device)
    def forward(self, state):
        x = torch.flatten(state, start_dim=1).to(self.device)
        x = self.input(x)
        x = F.relu(x)
        x = self.hidden(x)
        sigma = self.sigma(x)
        mu = self.mu(x)
        # reshape output as the same shape as action require.
        mu = torch.reshape(mu, (state.shape[0],) + self.action_shape)
        sigma = torch.reshape(sigma, (state.shape[0],) + self.action_shape)
        return mu, sigma


class TDActorNetwork(GaussianPolicyNetwork):
    def __init__(self, lr, state_shape, action_shape, device, 
        n_hiddens=2, hidden_size=256, name='actor_gauss', chkpt='./temp') -> None:
        super().__init__(lr, state_shape, action_shape, 
            device, n_hiddens=n_hiddens, hidden_size=hidden_size, 
            name=name, chkpt=chkpt)
    def act(self, state):
        pass
    def evaluate(self, state, action):
        pass


#----------------------------------------------------------------------------
# Main function test
#----------------------------------------------------------------------------
if __name__ == '__main__':
    action_value_net = ActionValueNetwork(
        lr=1e-3,
        state_shape=(2,3),
        action_shape=(2,2),
        device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    state_value_net = StateValueNetwork(
        lr=1e-3,
        state_shape=(2,3),
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        n_hiddens=2
    )
    gauss = GaussianPolicyNetwork(
        lr=1e-3,
        state_shape=(2,3),
        action_shape=(2,2),
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    # for param in state_value_net.parameters():
    #     print(param.shape)
    state = torch.rand((5,2,3))
    action = torch.rand((5,2,2))
    # x = action_value_net.forward(state, action)
    # print(x, x.shape)
    # y = state_value_net.forward(state)
    # print(y, y.shape)
    mu, sigma = gauss.forward(state)
    print(mu.shape, sigma.shape)
    # mu_ = torch.reshape(mu, (5,) + (2,2))
    # print(mu_.shape)
