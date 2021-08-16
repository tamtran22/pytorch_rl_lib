import torch
import torch.nn as nn
import torch.optim as optim
from network.base_network import BaseNetwork
from utils.utils import prod



#----------------------------------------------------------------------------
# Distributions
#----------------------------------------------------------------------------
class CategoricalDistributionNetwork(BaseNetwork):
    def __init__(self, lr, input_shape, n_actions, device, n_hiddens=2,
            hidden_size=256, name='actor_categorical', 
            chkpt='./tmp') -> None:
        super().__init__(device, name, chkpt)
        self.input_shape = input_shape
        self.output_shape = (n_actions,)

        # Flatten input tensor
        self.create_input_layer(self.input_shape, hidden_size)
        self.create_hidden_layers(n_hiddens, hidden_size)
        self.create_output_layer(hidden_size, self.output_shape)

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
        x = torch.sigmoid(x) # Probability of each action.
        return x
        



class GaussianDistributionNetwork(BaseNetwork):
    def __init__(self, lr, input_shape, output_shape, device, n_hiddens=2,
            hidden_size=256, reshape_output=False, name='actor_gauss',
            chkpt='./tmp') -> None:
        super().__init__(device, name, chkpt)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.reshape_output = reshape_output

        # Flaten input tensor
        self.create_input_layer(self.input_shape, hidden_size)
        self.create_hidden_layers(n_hiddens, hidden_size)
        self.sigma = nn.Sequential(
            nn.Linear(hidden_size, prod(self.action_shape))
            # nn.Sigmoid()
        )
        self.mu = nn.Linear(hidden_size, prod(self.action_shape))

        self.optimizer = optim.Adam(
            self.parameters(), 
            lr=lr
        )
        self.to(self.device)
    def forward(self, state):
        x = torch.flatten(state, start_dim=1).to(self.device)
        x = self.input(x)
        x = self.hidden(x)
        sigma = self.sigma(x)
        mu = self.mu(x)
        # reshape output as the same shape as action require.
        if self.reshape_output:
            mu = torch.reshape(mu, (state.shape[0],) + self.output_shape)
            sigma = torch.reshape(sigma, (state.shape[0],) + self.output_shape)
        return mu, sigma




        



#----------------------------------------------------------------------------
# Main function test
#----------------------------------------------------------------------------
if __name__ == '__main__':
    # action_value_net = ActionValueNetwork(
    #     lr=1e-3,
    #     state_shape=(2,3),
    #     action_shape=(2,2),
    #     device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # )
    # state_value_net = StateValueNetwork(
    #     lr=1e-3,
    #     state_shape=(2,3),
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    #     n_hiddens=2
    # )
    gauss = GaussianDistributionNetwork(
        lr=1e-3,
        state_shape=(2,3),
        action_shape=(2,2),
        reshape_action=True,
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
